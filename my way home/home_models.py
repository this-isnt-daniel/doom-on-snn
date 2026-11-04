"""Matched CNN and snnTorch rate-coded navigation policies for Sample Factory."""
import torch
from torch import nn
from sample_factory.model.encoder import Encoder
from sample_factory.model.actor_critic import ActorCriticSharedWeights
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.algo.utils.action_distributions import get_action_distribution


class TriangularSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage):
        ctx.save_for_backward(voltage)
        return (voltage > 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, grad):
        (voltage,) = ctx.saved_tensors
        return grad * (1-voltage.abs()).clamp(min=0)


def rate_step(probability, accumulator):
    """Deterministic pulse-density coding; at most one spike/pixel/internal step."""
    accumulator = accumulator + probability
    spike = (accumulator >= 1).to(probability.dtype)
    return spike, accumulator-spike


class ImageEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        self.spiking = cfg.home_model == "snn"
        self.T = int(cfg.home_T)
        if self.T < 1: raise ValueError("home_T must be positive")
        self.layers = nn.ModuleList([
            nn.Conv2d(obs_space["obs"].shape[0],32,8,stride=4),
            nn.Conv2d(32,64,4,stride=2), nn.Conv2d(64,128,3,stride=2),
            nn.Linear(2304,512)])
        if not self.spiking:self.relu = nn.ReLU()
        self.register_buffer("gains", torch.tensor([float(v) for v in cfg.home_gains.split(",")]))
        if self.gains.numel() != 5: raise ValueError("Five gains required, including output")
        if self.spiking:
            import snntorch as snn
            if not 0 <= cfg.home_beta <= 1: raise ValueError("beta must be in [0,1]")
            self.policy_synapses = nn.Linear(512,6)
            self.neurons = nn.ModuleList([
                snn.Leaky(beta=cfg.home_beta, threshold=1., spike_grad=TriangularSpike.apply,
                          reset_mechanism="subtract", reset_delay=False,
                          learn_beta=False, learn_threshold=False, init_hidden=False)
                for _ in range(5)])
            # Explicit states are managed here; do not let snnTorch's optional
            # global instance registry retain discarded calibration/worker models.
            snn.SpikingNeuron.instances[:] = [n for n in snn.SpikingNeuron.instances
                                              if not any(n is own for own in self.neurons)]

    def get_out_size(self): return 518 if self.spiking else 512

    def forward(self, obs): return self.run(obs["obs"])[0]

    def run(self, image, collect=False):
        if not self.spiking:
            x=image
            for k,layer in enumerate(self.layers):
                if k==3: x=x.flatten(1)
                x=self.relu(layer(x))
            return x, None
        # No Poisson resampling: identical observations give identical PPO logits.
        p=image.clamp(0,1)
        accumulator=torch.full_like(p,.5)
        mem=[None]*5; hidden_sum=0; output_sum=0
        spike_totals=[] if collect else None
        totals=[0.]*6
        try:
            for _ in range(self.T):
                x,accumulator=rate_step(p,accumulator)
                if collect: totals[0]=totals[0]+x.detach().flatten(1).sum(1)
                for k,layer in enumerate(self.layers):
                    if k==3: x=x.flatten(1)
                    current=layer(x)*self.gains[k]
                    if mem[k] is None: mem[k]=torch.full_like(current,.5)
                    x,mem[k]=self.neurons[k](current,mem[k])
                    if collect: totals[k+1]=totals[k+1]+x.detach().flatten(1).sum(1)
                hidden_sum=hidden_sum+x
                current=self.policy_synapses(x)*self.gains[4]
                if mem[4] is None: mem[4]=torch.full_like(current,.5)
                output,mem[4]=self.neurons[4](current,mem[4])
                output_sum=output_sum+output
                if collect: totals[5]=totals[5]+output.detach().sum(1)
            # First 512 values are critic features; last six are output firing rates.
            encoded=torch.cat([hidden_sum/self.T,output_sum/self.T],dim=-1)
            if collect: spike_totals=torch.stack(totals,dim=1)
            return encoded,spike_totals
        finally:
            # snnTorch stores its last membrane internally even with explicit mem.
            # Release graph references; every new observation starts at v_init=.5.
            for neuron in self.neurons:
                neuron.mem=torch.empty(0,device=image.device)
                neuron.reset=torch.empty(0,device=image.device)


def encoder_factory(cfg, obs_space): return ImageEncoder(cfg, obs_space)


class CountDecoder(nn.Module):
    """Fixed decoding only: no learned parameters after output spikes."""
    def __init__(self, scale):
        super().__init__()
        if scale <= 0: raise ValueError("Output-logit scale must be positive")
        self.register_buffer("scale",torch.tensor(float(scale)))

    def forward(self, rates, action_mask=None):
        logits=rates*self.scale
        return logits,get_action_distribution(self.action_space,raw_logits=logits,action_mask=action_mask)


class SpikingActorCritic(ActorCriticSharedWeights):
    def get_action_parameterization(self, decoder_output_size):
        decoder=CountDecoder(self.cfg.home_logit_scale)
        decoder.action_space=self.action_space
        return decoder

    def __init__(self, model_factory, obs_space, action_space, cfg):
        if cfg.use_rnn or cfg.decoder_mlp_layers:
            raise ValueError("This feedforward spiking policy requires use_rnn=False and an empty decoder")
        super().__init__(model_factory,obs_space,action_space,cfg)
        self.critic_linear=nn.Linear(512,1)
        self.initialize_weights(self.critic_linear)

    def forward_tail(self, core_output, values_only, sample_actions, action_mask=None):
        result=TensorDict(values=self.critic_linear(core_output[:,:512]).squeeze())
        if values_only: return result
        logits,self.last_action_distribution=self.action_parameterization(core_output[:,512:],action_mask)
        result["action_logits"]=logits
        self._maybe_sample_actions(sample_actions,result)
        return result


def actor_factory(cfg, obs_space, action_space):
    model=SpikingActorCritic(global_model_factory(),obs_space,action_space,cfg)
    if cfg.home_init_state:
        model.load_state_dict(torch.load(cfg.home_init_state,map_location="cpu",weights_only=False),strict=True)
    return model


def register_models(model):
    global_model_factory().register_encoder_factory(encoder_factory)
    if model=="snn": global_model_factory().register_actor_critic_factory(actor_factory)


def add_args(parser):
    parser.add_argument("--home_model",choices=["ann","snn"],default="ann")
    parser.add_argument("--home_T",type=int,default=8)
    parser.add_argument("--home_beta",type=float,default=.9)
    parser.add_argument("--home_logit_scale",type=float,default=8.)
    parser.add_argument("--home_gains",type=str,default="1,1,1,1,1")
    parser.add_argument("--home_stack",type=int,default=1)
    parser.add_argument("--home_init_state",type=str,default="")


@torch.no_grad()
def calibrate(model, frames, target=.15, iterations=14):
    """Calibrate all five spiking layers on the actual initialized actor's weights."""
    encoder=model.encoder
    if not encoder.spiking: raise ValueError("Calibration is SNN-only")
    model.eval()
    encoder.gains.fill_(1.)
    neurons=torch.tensor([32*17*31,64*7*14,128*3*6,512,6],device=frames.device)
    def rates():
        _,totals=encoder.run(frames,collect=True)
        return totals[:,1:].mean(0)/(neurons*encoder.T)
    for k in range(5):
        lo,hi=-3.,3.
        for _ in range(iterations):
            mid=(lo+hi)/2;encoder.gains[k]=10**mid
            if rates()[k].item()<target:lo=mid
            else:hi=mid
        encoder.gains[k]=10**((lo+hi)/2)
    return encoder.gains.cpu().tolist(),rates().cpu().tolist()

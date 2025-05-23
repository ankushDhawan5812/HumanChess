import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class Diffusion:
    def __init__(self, actions_space, elo_min, elo_max, bucket_size, d_emb, betas):
        self.actions_space = actions_space
        self.betas = torch.tensor(betas)
        self.alpha = torch.cumprod(1 - self.betas, dim=0)
    
    def diffuse(self, batch):
        state_tokens = [idx['s_tokens'] for idx in batch] # initial ids
        future_tokens = [idx['f_tokens'] for idx in batch] # future states and actions
        elo_tokens = torch.stack([idx['elo_idx_float'] for idx in batch]) # elo index to prepend

        state_tokens  = pad_sequence(state_tokens, batch_first=True, padding_value=0)
        future_tokens  = pad_sequence(future_tokens, batch_first=True, padding_value=0)
        batch_size = future_tokens.shape[0]
        max_length = future_tokens.shape[1]

        sample_t = torch.randint(0, len(self.betas), (batch_size,)) # need the random noise idx

        true_samples = F.one_hot(future_tokens, num_classes=self.actions_space).float() #this will be batch_siz x max_length x 1968
        cur_alpha = self.alpha[sample_t].unsqueeze(-1).unsqueeze(-1)
        noise_dist = cur_alpha * true_samples + (1 - cur_alpha) * (1/self.actions_space)
        x_noise = torch.multinomial(noise_dist.reshape((batch_size * max_length, -1)), num_samples=1).reshape(batch_size, max_length)

        return {
            "s_tokens": state_tokens,
            "x_0": future_tokens,
            "elo_idx_float": elo_tokens,
            "x_t": x_noise,
            "t": sample_t,
        }



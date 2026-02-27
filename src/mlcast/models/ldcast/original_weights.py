import torch
import re

def convert_original_weights(ldm_weights_fn):
    '''
    returns the original weights of the denoiser and the conditioner from the way they were saved originally
    at the moment, the ema scope is not taken into account
    the unmatched_keys are the ema keys and the buffer keys for the schedule (at the moment)
    '''
    ldm_state_dict = torch.load(ldm_weights_fn)

    # track unmatched keys
    unmatched_keys = list(ldm_state_dict.keys())
    
    # remove the weights of the autoencoder
    for k in unmatched_keys.copy():
        if k.startswith('autoencoder.') or k.startswith('context_encoder.autoencoder.'):
            unmatched_keys.remove(k)

    # extract the keys of the denoiser (it was called 'model' in the original code)
    denoiser_state_dict = {}
    for k in unmatched_keys.copy():
        if k.startswith('model.'):
            new_key = k.replace('model.', '')
            denoiser_state_dict[new_key] = ldm_state_dict[k]
            unmatched_keys.remove(k)

    denoiser_buffers_keys = ['betas', 'alphas_cumprod', 'alphas_cumprod_prev', 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod']
    for k in unmatched_keys.copy():
        if k in denoiser_buffers_keys:
            denoiser_state_dict[k] = ldm_state_dict[k]
            unmatched_keys.remove(k)
    
    # extract the keys of the conditioner (it was called 'context_encoder' in the original code)
    conditioner_state_dict = {}
    for k in unmatched_keys.copy():
        if k.startswith('context_encoder.'):
                new_key = k.replace('context_encoder.', '')
                conditioner_state_dict[new_key] = ldm_state_dict[k]
                unmatched_keys.remove(k)

    # proj, temporal_transformer and analysis were lists with one only element, I simplified this
    # the keys have to be adapted
    new_conditioner_state_dict = {}
    for k, v in conditioner_state_dict.items():
        new_key = k
        if k.startswith('proj.0.'):
            new_key = k.replace('proj.0.', 'proj.')
        if k.startswith('temporal_transformer.0.'):
            new_key = k.replace('temporal_transformer.0.', 'temporal_transformer.')
        if k.startswith('analysis.0.'):
            new_key = k.replace('analysis.0.', 'analysis.')
        new_conditioner_state_dict[new_key] = v
    conditioner_state_dict = new_conditioner_state_dict

    ema = {}
    for k in unmatched_keys.copy():
        if k.startswith('model_ema.'):
            new_key = restore_name(k.replace('model_ema.', ''))
            ema[new_key] = ldm_state_dict[k]
            unmatched_keys.remove(k)
    
    # create dict with unmatched keys
    unmatched = {key: ldm_state_dict[key] for key in unmatched_keys}
    
    return {'denoiser': denoiser_state_dict,
            'conditioner': conditioner_state_dict,
            'ema': ema,
            'unmatched': unmatched}

def restore_name(s):
    '''for the EMA, all the dots were removed from the parameters names in the original code, so they should be added again to match during swapping'''
    # add dots before and after every digit
    res = re.sub(r'(\d)', r'.\1.', s)
    # if the digit was in 'fc1', 'fc2', it should not be preceded by a dot
    res = res.replace('fc.1', 'fc1')
    res = res.replace('fc.2', 'fc2')
    # same, but there should be in addition a dot before w1, w2, b1 and b2
    res = res.replace('w.1.', '.w1')
    res = res.replace('w.2.', '.w2')
    res = res.replace('b.1.', '.b1')
    res = res.replace('b.2.', '.b2')
    # add a dot before each 'weights' and each 'bias'
    res = res.replace('weight', '.weight').replace('bias', '.bias')
    # add dot after mlp
    res = res.replace('mlp', 'mlp.')
    # if two dots are inserted, replace them by one (happens if two digits follow each other, or if a digit is followed by b or w)
    res = res.replace('..', '.')
    return res
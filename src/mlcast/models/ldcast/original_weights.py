import torch

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
    
    # extract the keys of the conditioner (it was called 'context_encoder' in the original code)
    conditioner_state_dict = {}
    for k in unmatched_keys.copy():
        if k.startswith('context_encoder.'):
                new_key = k.replace('context_encoder.', '')
                conditioner_state_dict[new_key] = ldm_state_dict[k]
                unmatched_keys.remove(k)

    # proj, temporal_transformer and analysis were lists one only element, I simplified this
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

    # create dict with unmatched keys
    unmatched = {key: ldm_state_dict[key] for key in unmatched_keys}
    
    return {'denoiser_state_dict': denoiser_state_dict,
            'conditioner_state_dict': conditioner_state_dict,
            'unmatched': unmatched}

def check_saved_buffers(d, ldm):
    '''
    checks that the buffers saved in ldm are the same than the ones in d (which is a dict containing these values)
    returns the unmatched elements in d
    '''

    unmatched_keys = list(d.keys())
    
    for buffer in ldm.named_buffers():
        name, value = buffer
        assert (value == d[name].to(value.device)).all()
        unmatched_keys.remove(name)

    # create dict with unmatched keys
    unmatched = {key: d[key] for key in unmatched_keys}
    
    return unmatched
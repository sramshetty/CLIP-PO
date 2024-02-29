

# Referencing: https://github.com/mlfoundations/patching/blob/main/src/patch.py
def patch_model(ref_model, tuned_model, alpha=0.3):

    theta_0 = {k: v.clone() for k, v in ref_model.state_dict().items()}
    theta_1 = {k: v.clone() for k, v in tuned_model.state_dict().items()}
    del ref_model

    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())

    # interpolate between all weights in the checkpoints
    theta = {
        key: (1-alpha) * theta_0[key] + alpha * theta_1[key]
        for key in theta_0.keys()
    }

    # update the model (in-place) acccording to the new weights
    tuned_model.load_state_dict(theta)

    return tuned_model

"""Validation-only stopping: a resource limit is never a convergence claim."""
import math


def record_validation(state, epoch, score, policy):
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Validation AP must be finite and lie in [0,1]')
    if epoch != len(state.get('validation', [])) + 1:
        raise ValueError('Validation epochs must be contiguous and recorded once')
    rows = state.setdefault('validation', [])
    best = state.get('best_ap')
    selected = best is None or score > best
    if selected:
        state.update(best_ap=score, best_epoch=epoch)
    anchor = state.get('significant_ap')
    if anchor is None or score > anchor + policy['min_delta']:
        state.update(significant_ap=score, bad_epochs=0)
    else:
        state['bad_epochs'] += 1
    rows.append({'epoch': epoch, 'video_max_ap': score, 'selected_best': selected,
                 'bad_epochs': state['bad_epochs']})
    stopped = epoch >= policy['min_epochs'] and state['bad_epochs'] >= policy['patience']
    if stopped:
        state.update(stopped=True, stop_reason='validation_plateau', stopped_epoch=epoch)
    return selected, stopped


def convergence_summary(arms, epoch_limit):
    if all(s.get('stop_reason') == 'validation_plateau' for s in arms.values()):
        return 'complete'
    if max((len(s.get('validation', [])) for s in arms.values()), default=0) >= epoch_limit:
        return 'needs_extension'
    return 'running'

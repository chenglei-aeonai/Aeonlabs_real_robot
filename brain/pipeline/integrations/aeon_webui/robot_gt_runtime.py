"""Launch H3 with custom-frame validity fixes, without editing shared runtime files."""
from pathlib import Path
import sys
from types import ModuleType


def custom_validity(base, valid):
    """Use bundle L/R validity across all frames and all 138 v2 channels."""
    import torch
    from h3_joint.representations import S_LTSL, S_RTSL, S_LROT, S_RROT, S_LFING, S_RFING
    result = dict(base)
    raw = base['skel_raw'].clone()
    if raw.ndim != 2 or raw.shape[-1] != 138:
        raise ValueError('Robot GT runtime requires v2_abspose (T,138).')
    mask = torch.zeros_like(raw, dtype=torch.bool)
    for hand, slices in enumerate(((S_LTSL, S_LROT, S_LFING), (S_RTSL, S_RROT, S_RFING))):
        for part in slices:
            mask[:, part] = bool(valid[hand])
            if not valid[hand]:
                raw[:, part] = 0
    result.update(skel_raw=raw, chan_valid=mask)
    return result


def patch_source(source):
    replacements = [
        ('        encoded = self.encode_continuation(b["frame"], prompt)',
         '        from robot_gt_runtime import custom_validity\n'
         '        base_cond = custom_validity(base_cond, b["valid"])\n'
         '        encoded = self.encode_continuation(b["frame"], prompt)'),
        ('        J_pred, _ = decode_v2_abspose(skel_raw)',
         '        J_pred, wrist_R_pred = decode_v2_abspose(skel_raw)'),
        ('                 K=cam_K.detach().float().cpu().numpy(), fps=H3_FPS)',
         '                 K=cam_K.detach().float().cpu().numpy(), fps=H3_FPS,\n'
         '                 wrist_R=wrist_R_pred[:T].detach().float().cpu().numpy(),\n'
         '                 valid=(self.custom_frames[cid]["valid"]\n'
         '                        if cid in self.custom_frames else\n'
         '                        item_cond["chan_valid"][:T, [0, 3]].detach().cpu().numpy()))'),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Unsupported H3 runtime source; refusing an ambiguous validity patch: ' + old.strip())
        source = source.replace(old, new)
    compile(source, '<robot-gt-runtime>', 'exec')
    return source


def main():
    path = Path(sys.argv.pop(1)).resolve()
    source = patch_source(path.read_text())
    if sys.argv[1:] == ['--check']:
        print('Robot GT runtime: custom validity patch verified; shared files unchanged.')
        return
    sys.argv[0] = str(path)
    print('Robot GT runtime: bundle L/R validity overrides demo template; exports wrist_R and valid.', flush=True)
    module = ModuleType('__main__')
    module.__file__ = str(path)
    sys.modules['__main__'] = module
    exec(compile(source, str(path), 'exec'), module.__dict__)


if __name__ == '__main__':
    main()

"""Install the GT upload route into an explicitly selected aeon-webui checkout."""
import argparse
from pathlib import Path
import shutil

ANCHOR = '# Static mounts LAST — "/" would otherwise shadow the API routes.'
INSERT = 'from robot_gt_bridge import install_routes as _install_robot_gt\n_install_robot_gt(app, globals())\n\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    args = parser.parse_args()
    target = args.checkout / 'app.py'
    helper = args.checkout / 'robot_gt_bridge.py'
    text = target.read_text()
    if helper.exists() or INSERT in text or text.count(ANCHOR) != 1:
        raise ValueError('Unsupported source or GT bridge already installed; refusing overwrite.')
    result = text.replace(ANCHOR, INSERT + ANCHOR, 1)
    compile(result, str(target), 'exec')
    shutil.copyfile(Path(__file__).with_name('robot_gt_bridge.py'), helper)
    target.write_text(result)
    print('Installed GT frame upload. Restart the isolated aeon-webui UI; the existing H3 custom-frame runtime must already be configured.')


if __name__ == '__main__':main()

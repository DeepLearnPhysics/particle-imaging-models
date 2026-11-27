import ast

from pimm.utils.config import Config


def test_pretty_text_roundtrips_shell_quoted_string(tmp_path):
    launch_command = (
        "pimm submit -- scheduler.max_lr='[0.000075,0.0000075]' " 'notes="fresh run"'
    )
    source = tmp_path / "source.py"
    source.write_text("launch_command = None\n")
    cfg = Config.fromfile(str(source))
    cfg.launch_command = launch_command

    text = cfg.pretty_text
    ast.parse(text)

    path = tmp_path / "config.py"
    cfg.dump(str(path))
    restored = Config.fromfile(str(path))
    assert restored.launch_command == launch_command

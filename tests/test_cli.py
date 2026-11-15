import pytest

from lunar_mpc.cli import main


@pytest.mark.parametrize(
    "arguments",
    [
        ["run", "--seed", "-1"],
        ["evaluate", "--episodes", "0"],
        ["evaluate", "--render"],
    ],
)
def test_invalid_command_options_fail_explicitly(arguments, capsys):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("is_directory", [True, False])
def test_output_cannot_overwrite_existing_results(tmp_path, is_directory):
    output = tmp_path / "existing"
    if is_directory:
        output.mkdir()
        original = output / "summary.json"
    else:
        original = output
    original.write_text("preserve me")
    with pytest.raises(SystemExit) as error:
        main(["run", "--output", str(output)])
    assert error.value.code == 2
    assert original.read_text() == "preserve me"

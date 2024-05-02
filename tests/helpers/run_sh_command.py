from typing import List

import pytest

from tests.helpers.package_available import _SH_AVAILABLE

if _SH_AVAILABLE:
    import sh


def run_sh_command(command: List[str]) -> None:
    """Default method for executing shell commands with `pytest` and `sh` package.

    :param command: A list of shell commands as strings.
    """
    msg_stderr_output = None
    try:
        sh.python(command)
    except sh.ErrorReturnCode as e:
        msg_stderr_output = e.stderr.decode()
    if msg_stderr_output:
        pytest.fail(reason=msg_stderr_output)  # msg argument deprecated (use reason instead)

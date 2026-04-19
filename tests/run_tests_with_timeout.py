import os
import subprocess
import sys


def run_with_timeout(command, timeout_sec):
    print(f"Running command with {timeout_sec}s timeout: {' '.join(command)}")
    try:
        # Use subprocess.run with timeout
        result = subprocess.run(
            command,
            timeout=timeout_sec,
            capture_output=False,  # Show output in real-time
            text=True,
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"\n❌ ERROR: Command timed out after {timeout_sec} seconds!")
        return 124  # Standard timeout exit code
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return 1


if __name__ == "__main__":
    # Example: python run_tests_with_timeout.py pytest tests/smoke/test_api.py -m smoke
    cmd = sys.argv[1:]
    if not cmd:
        cmd = ["uv", "run", "pytest", "tests/smoke/test_api.py", "-m", "smoke"]

    sys.exit(run_with_timeout(cmd, 60))

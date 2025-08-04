import asyncio
import sys

async def stream_subprocess(command: list[str], description: str) -> bool:
    """
    Runs a command asynchronously and streams its stdout/stderr to the console.
    This prevents blocking and allows for real-time log monitoring.
    """
    print(f"--> Running command: {' '.join(command)}")
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    async def log_stream(stream, prefix):
        while True:
            line = await stream.readline()
            if not line:
                break
            print(f"{prefix} {line.decode().strip()}", flush=True)

    # Concurrently log stdout and stderr
    await asyncio.gather(
        log_stream(process.stdout, f"[{description}-stdout]"),
        log_stream(process.stderr, f"[{description}-stderr]")
    )

    await process.wait()
    if process.returncode != 0:
        print(f"--> Command for '{description}' failed with exit code {process.returncode}")
        return False
        
    print(f"--> Command for '{description}' finished successfully.")
    return True
def run(input, exception_on_failure=True, silent=False):
    try:
        import subprocess
        print(input)
        program_output = subprocess.check_output(f"{input}", shell=True, universal_newlines=True,
                                                 stderr=subprocess.STDOUT)
    except Exception as e:
        program_output = e.output
        if exception_on_failure:
            if not silent:
                print(program_output)
            raise e

    if not silent:
        print(program_output)

    return program_output.strip()
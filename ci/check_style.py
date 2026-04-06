#!/usr/bin/env python3
"""
Style checker and formatter for the Athena project.
Checks and fixes code style for C++ and Python files.

Configuration:
- .clang-format: C++ code formatting rules (based on Google style)
- .clang-tidy: C++ static analysis and linting rules
- Uses black for Python formatting and pylint for Python linting
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


class StyleChecker:
    """Main style checker class."""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, project_root: str, fix: bool = False, verbose: bool = False):
        """
        Initialize the style checker.

        Args:
            project_root: Root directory of the project
            fix: Whether to fix style issues automatically
            verbose: Verbose output
        """
        self.project_root = Path(project_root).resolve()
        self.fix = fix
        self.verbose = verbose
        self.cpp_extensions = {".hpp", ".cpp", ".h", ".cu", ".cuh"}
        self.python_extensions = {".py"}
        self.shell_extensions = {".sh", ".bash"}
        self.cmake_files = {"CMakeLists.txt"}
        self.errors = []
        self.warnings = []
        # Change to project root to ensure tools find config files
        self.original_cwd = os.getcwd()
        os.chdir(self.project_root)

    def log(self, message: str, level: str = "INFO") -> None:
        """Log messages with optional verbose output."""
        if level == "INFO" and self.verbose:
            print(f"[{level}] {message}")
        elif level != "INFO":
            print(f"[{level}] {message}")

    def find_files(self, extensions: set = None, names: set = None) -> list:
        """Find all files with given extensions or names."""
        files = []
        exclude_dirs = {".git", "build", ".venv", "__pycache__", "ext"}
        extensions = extensions or set()
        names = names or set()

        for root, dirs, filenames in os.walk(self.project_root):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for filename in filenames:
                if (
                    any(filename.endswith(ext) for ext in extensions)
                    or filename in names
                ):
                    files.append(Path(root) / filename)

        return sorted(files)

    def check_cpp_style(self) -> int:
        """Check C++ code style using clang-format."""
        self.log("Checking C++ code style with clang-format...")
        cpp_files = self.find_files(self.cpp_extensions)

        if not cpp_files:
            self.log("No C++ files found.")
            return 0

        errors_found = 0

        for cpp_file in cpp_files:
            self.log(f"Checking {cpp_file.relative_to(self.project_root)}")

            try:
                if self.fix:
                    # Fix the formatting directly
                    subprocess.run(
                        [
                            "clang-format",
                            "-i",
                            "--style=file",
                            str(cpp_file),
                        ],
                        capture_output=True,
                        check=False,
                        timeout=10,
                    )
                    self.log(
                        f"Fixed formatting in {cpp_file.relative_to(self.project_root)}",
                        "FIXED",
                    )
                else:
                    # Check if file is formatted
                    result = subprocess.run(
                        [
                            "clang-format",
                            "--dry-run",
                            "-Werror",
                            "--style=file",
                            str(cpp_file),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=10,
                    )

                    if result.returncode != 0:
                        self.errors.append(
                            f"C++ formatting issue in {cpp_file.relative_to(self.project_root)}"
                        )
                        errors_found += 1

            except FileNotFoundError:
                self.warnings.append(
                    "clang-format not found. Install with: sudo apt-get install clang-format"
                )
                return -1
            except subprocess.TimeoutExpired:
                self.errors.append(f"Timeout checking {cpp_file}")
                errors_found += 1

        return errors_found

    def check_cpp_lint(self) -> int:
        """Check C++ code quality using clang-tidy."""
        self.log("Checking C++ code quality with clang-tidy...")
        cpp_files = self.find_files(self.cpp_extensions)

        if not cpp_files:
            return 0

        errors_found = 0

        for cpp_file in cpp_files:
            self.log(f"Linting {cpp_file.relative_to(self.project_root)}")

            try:
                config_path = self.project_root / ".clang-tidy"
                result = subprocess.run(
                    [
                        "clang-tidy",
                        str(cpp_file),
                        "--config-file=" + str(config_path),
                        "--",
                        "-I",
                        str(self.project_root / "app"),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )

                if result.stdout and "error:" in result.stdout.lower():
                    self.warnings.append(
                        f"clang-tidy issues in {cpp_file.relative_to(self.project_root)}"
                    )
                    if self.verbose:
                        self.log(result.stdout, "LINT")

            except FileNotFoundError:
                if errors_found == 0:  # Only warn once
                    msg = "clang-tidy not found. Install with: "
                    msg += "sudo apt-get install clang-tools clang-format"
                    self.warnings.append(msg)
                return errors_found
            except subprocess.TimeoutExpired:
                self.warnings.append(f"clang-tidy timeout on {cpp_file}")

        return errors_found

    def check_shell_style(self) -> int:
        """Check shell script style using shfmt."""
        self.log("Checking shell script style with shfmt...")
        shell_files = self.find_files(self.shell_extensions)

        if not shell_files:
            self.log("No shell scripts found.")
            return 0

        errors_found = 0

        for shell_file in shell_files:
            self.log(f"Checking {shell_file.relative_to(self.project_root)}")

            try:
                if self.fix:
                    # Fix shell script formatting
                    subprocess.run(
                        ["shfmt", "-i", "4", "-w", str(shell_file)],
                        capture_output=True,
                        check=False,
                        timeout=10,
                    )
                    self.log(
                        f"Fixed shell format in {shell_file.relative_to(self.project_root)}",
                        "FIXED",
                    )
                else:
                    # Check shell script formatting
                    result = subprocess.run(
                        ["shfmt", "-i", "4", "-d", str(shell_file)],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=10,
                    )

                    if result.returncode != 0:
                        msg = (
                            f"Shell script format issue in "
                            f"{shell_file.relative_to(self.project_root)}"
                        )
                        self.errors.append(msg)
                        errors_found += 1

            except FileNotFoundError:
                self.warnings.append(
                    "shfmt not found. Install with: sudo apt-get install shfmt"
                )
            except subprocess.TimeoutExpired:
                self.errors.append(f"Timeout checking {shell_file}")
                errors_found += 1

        return errors_found

    def check_cmake_style(self) -> int:
        """Check CMakeLists.txt files."""
        self.log("Checking CMakeLists.txt files...")
        cmake_files = self.find_files(names=self.cmake_files)

        if not cmake_files:
            self.log("No CMakeLists.txt files found.")
            return 0

        errors_found = 0

        for cmake_file in cmake_files:
            self.log(f"Checking {cmake_file.relative_to(self.project_root)}")

            try:
                if self.fix:
                    # Format CMakeLists with cmake-format
                    subprocess.run(
                        ["cmake-format", "-i", str(cmake_file)],
                        capture_output=True,
                        check=False,
                        timeout=10,
                    )
                    self.log(
                        f"Fixed cmake format in {cmake_file.relative_to(self.project_root)}",
                        "FIXED",
                    )
                else:
                    # Check cmake formatting
                    result = subprocess.run(
                        ["cmake-format", "--check", str(cmake_file)],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=10,
                    )

                    if result.returncode != 0:
                        self.errors.append(
                            f"CMake format issue in {cmake_file.relative_to(self.project_root)}"
                        )
                        errors_found += 1

            except FileNotFoundError:
                self.warnings.append(
                    "cmake-format not found. Install with: pip install cmake-format"
                )
            except subprocess.TimeoutExpired:
                self.errors.append(f"Timeout checking {cmake_file}")
                errors_found += 1

        return errors_found

    def check_python_style(self) -> int:
        """Check Python code style using pylint and black."""
        self.log("Checking Python code style...")
        py_files = self.find_files(self.python_extensions)

        if not py_files:
            self.log("No Python files found.")
            return 0

        errors_found = 0

        for py_file in py_files:
            self.log(f"Checking {py_file.relative_to(self.project_root)}")

            if self.fix:
                # Try to auto-format with black
                try:
                    subprocess.run(
                        ["black", str(py_file)],
                        capture_output=True,
                        check=False,
                        timeout=10,
                    )
                    self.log(
                        f"Formatted {py_file.relative_to(self.project_root)}", "FIXED"
                    )
                except FileNotFoundError:
                    self.warnings.append(
                        "black not found. Install with: pip install black"
                    )
                except subprocess.TimeoutExpired:
                    self.errors.append(f"Timeout formatting {py_file}")
                    errors_found += 1

            # Check with pylint
            try:
                result = subprocess.run(
                    ["pylint", "--exit-zero", str(py_file)],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )
                if result.stdout:
                    self.log(result.stdout, "WARNING")
            except FileNotFoundError:
                self.warnings.append(
                    "pylint not found. Install with: pip install pylint"
                )

        return errors_found

    def check_line_endings(self) -> int:
        """Check for consistent line endings."""
        self.log("Checking line endings...")
        all_files = self.find_files(self.cpp_extensions | self.python_extensions)

        errors_found = 0

        for file_path in all_files:
            try:
                with open(file_path, "rb") as f:
                    content = f.read()

                    # Check for mixed line endings
                    has_crlf = b"\r\n" in content
                    has_lf = b"\n" in content and b"\r\n" not in content.replace(
                        b"\r\n", b""
                    )

                    if not (has_crlf or has_lf):
                        continue  # No line endings found, perhaps empty file

                    if has_crlf and has_lf:
                        self.errors.append(
                            f"Mixed line endings in {file_path.relative_to(self.project_root)}"
                        )
                        errors_found += 1

                        if self.fix:
                            # Fix to Unix line endings (LF)
                            content = content.replace(b"\r\n", b"\n")
                            with open(file_path, "wb") as fw:
                                fw.write(content)
                            self.log(
                                f"Fixed line endings in {file_path.relative_to(self.project_root)}",
                                "FIXED",
                            )

            except (IOError, OSError) as e:
                self.warnings.append(f"Error reading {file_path}: {e}")

        return errors_found

    def check_trailing_whitespace(self) -> int:
        """Check for trailing whitespace."""
        self.log("Checking for trailing whitespace...")
        all_files = self.find_files(self.cpp_extensions | self.python_extensions)

        errors_found = 0

        for file_path in all_files:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                if not any(line.rstrip() != line.rstrip("\n") for line in lines):
                    continue

                self.errors.append(
                    f"Trailing whitespace in {file_path.relative_to(self.project_root)}"
                )
                errors_found += 1

                if self.fix:
                    # Remove trailing whitespace
                    fixed_lines = [
                        line.rstrip() + "\n" if line.endswith("\n") else line.rstrip()
                        for line in lines
                    ]
                    with open(file_path, "w", encoding="utf-8") as fw:
                        fw.writelines(fixed_lines)
                    self.log(
                        f"Fixed trailing whitespace in {file_path.relative_to(self.project_root)}",
                        "FIXED",
                    )

            except (IOError, OSError) as e:
                self.warnings.append(f"Error reading {file_path}: {e}")

        return errors_found

    def run(self) -> int:
        """Run all style checks."""
        print("=" * 70)
        print("Code Style Checker")
        print("=" * 70)

        total_errors = 0

        # Run checks
        total_errors += self.check_cpp_style()
        total_errors += self.check_cpp_lint()
        total_errors += self.check_python_style()
        total_errors += self.check_shell_style()
        total_errors += self.check_cmake_style()
        total_errors += self.check_line_endings()
        total_errors += self.check_trailing_whitespace()

        # Print summary
        print("=" * 70)
        print("Style Check Summary")
        print("=" * 70)

        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")

        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  ✗ {error}")

        # Restore original working directory
        os.chdir(self.original_cwd)

        if total_errors == 0:
            print("\n✓ All style checks passed!")
            return 0
        print(f"\n✗ Found {total_errors} style issues.")
        if self.fix:
            print("  Issues have been automatically fixed.")
        else:
            print("  Run with --fix to automatically fix issues.")
        return 1


def main():
    """Main entry point.

    Uses configuration files:
    - .clang-format: C++ code formatting style
    - .clang-tidy: C++ code quality checks
    """
    parser = argparse.ArgumentParser(
        description="Check and fix code style for C++ and Python files."
    )
    parser.add_argument(
        "--fix", action="store_true", help="Automatically fix style issues"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parent.parent),
        help="Project root directory",
    )

    args = parser.parse_args()

    try:
        checker = StyleChecker(
            project_root=args.project_root, fix=args.fix, verbose=args.verbose
        )
        return checker.run()
    except (OSError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

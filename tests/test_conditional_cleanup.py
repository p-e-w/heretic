# SPDX-License-Identifier: AGPL-3.0-or-later
# Tests for conditional variable cleanup in run() — regression for issue #299.
#
# Issue: `del good_residuals, bad_residuals, analyzer` at line 466 of main.py
# crashes with UnboundLocalError when needs_full_residuals is False because
# `analyzer` is only assigned inside the `if needs_full_residuals:` branch.
#
# These tests verify the fix (initializing analyzer = None alongside
# good_residuals and bad_residuals) and audit the source to prevent regression.

import ast
import re
import textwrap
from pathlib import Path

import pytest

MAIN_PY = Path(__file__).resolve().parent.parent / "src" / "heretic" / "main.py"


@pytest.fixture(scope="module")
def source():
    return MAIN_PY.read_text()


@pytest.fixture(scope="module")
def tree(source):
    return ast.parse(source)


# ---------------------------------------------------------------------------
# 1. Source-level audits: ensure the fix is structurally correct
# ---------------------------------------------------------------------------

class TestAnalyzerInitialization:
    """Verify `analyzer` is initialized before the if/else branch."""

    def test_analyzer_initialized_to_none(self, source):
        """analyzer must be set to None before the needs_full_residuals branch."""
        lines = source.splitlines()
        init_line = None
        branch_line = None
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped == "analyzer = None":
                init_line = i
            if "needs_full_residuals" in stripped and stripped.startswith("if "):
                branch_line = i

        assert init_line is not None, (
            "analyzer = None initialization not found — "
            "it must be present before the conditional branch"
        )
        assert branch_line is not None, "needs_full_residuals branch not found"
        assert init_line < branch_line, (
            f"analyzer = None (line {init_line}) must appear before "
            f"the needs_full_residuals branch (line {branch_line})"
        )

    def test_all_del_targets_pre_initialized(self, source):
        """Every variable in `del good_residuals, bad_residuals, analyzer`
        must be initialized to None before the conditional branch."""
        # Find the del statement
        del_match = re.search(
            r"del\s+good_residuals\s*,\s*bad_residuals\s*,\s*analyzer", source
        )
        assert del_match, "del statement for residuals/analyzer not found"

        del_pos = del_match.start()

        # All three should be initialized before the del
        for var in ("good_residuals", "bad_residuals", "analyzer"):
            pattern = rf"^\s+{var}\s*=\s*None\s*$"
            match = re.search(pattern, source[:del_pos], re.MULTILINE)
            assert match is not None, (
                f"{var} = None not found before del statement — "
                f"this will cause UnboundLocalError when needs_full_residuals is False"
            )

    def test_initialization_ordering_matches_del_ordering(self, source):
        """The three None initializations should appear in the same order
        as they appear in the del statement, for readability."""
        positions = {}
        for var in ("good_residuals", "bad_residuals", "analyzer"):
            match = re.search(rf"^\s+{var}\s*=\s*None\s*$", source, re.MULTILINE)
            if match:
                positions[var] = match.start()

        assert len(positions) == 3, (
            f"Expected 3 None initializations, found {len(positions)}: "
            f"{list(positions.keys())}"
        )

        ordered = sorted(positions, key=positions.get)
        assert ordered == ["good_residuals", "bad_residuals", "analyzer"], (
            f"Initialization order {ordered} does not match "
            f"del statement order [good_residuals, bad_residuals, analyzer]"
        )


class TestDelStatementSafety:
    """Verify the del statement cannot raise UnboundLocalError."""

    def test_del_targets_all_have_unconditional_assignment(self, tree):
        """In run(), every target of the del statement on the cleanup line
        must have an unconditional assignment (not only inside an if branch)."""
        run_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run":
                run_func = node
                break

        assert run_func is not None, "run() function not found"

        # Find the del statement
        del_stmt = None
        for node in ast.walk(run_func):
            if isinstance(node, ast.Delete):
                target_names = [
                    t.id for t in node.targets if isinstance(t, ast.Name)
                ]
                if "analyzer" in target_names:
                    del_stmt = node
                    break

        assert del_stmt is not None, "del statement with analyzer not found"

        del_targets = [t.id for t in del_stmt.targets if isinstance(t, ast.Name)]

        # For each target, verify there's a top-level (non-nested-in-if) assignment
        # in the run() body before the del line
        for var in del_targets:
            found_unconditional = False
            for stmt in run_func.body:
                if stmt.lineno >= del_stmt.lineno:
                    break
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == var:
                            found_unconditional = True
                            break
            assert found_unconditional, (
                f"Variable '{var}' in del statement has no unconditional "
                f"assignment in run() before line {del_stmt.lineno} — "
                f"this will crash when needs_full_residuals is False"
            )


# ---------------------------------------------------------------------------
# 2. Behavioral simulation: verify cleanup doesn't crash
# ---------------------------------------------------------------------------

class TestCleanupBehavior:
    """Simulate the cleanup pattern from run() to verify it doesn't crash."""

    def test_cleanup_without_full_residuals(self):
        """Simulates the else branch (needs_full_residuals=False) followed
        by the del statement. This is the exact scenario from issue #299."""
        # Initialization (the fix)
        good_residuals = None
        bad_residuals = None
        analyzer = None

        needs_full_residuals = False

        if needs_full_residuals:
            # Would create Analyzer here in real code
            analyzer = "would be Analyzer instance"
            good_residuals = "would be tensor"
            bad_residuals = "would be tensor"
        else:
            # Only means are computed, no analyzer
            pass

        # This is the line that crashed before the fix
        del good_residuals, bad_residuals, analyzer

    def test_cleanup_with_full_residuals(self):
        """Simulates the if branch (needs_full_residuals=True) followed
        by the del statement. Should work in both fixed and unfixed code."""
        good_residuals = None
        bad_residuals = None
        analyzer = None

        needs_full_residuals = True

        if needs_full_residuals:
            good_residuals = "tensor"
            bad_residuals = "tensor"
            analyzer = "Analyzer instance"
        else:
            pass

        del good_residuals, bad_residuals, analyzer

    def test_cleanup_crash_without_fix(self):
        """Demonstrates that WITHOUT the fix, the else branch crashes.
        This test documents the original bug."""
        good_residuals = None
        bad_residuals = None
        # NOTE: no `analyzer = None` here — simulates the original bug

        needs_full_residuals = False

        if needs_full_residuals:
            analyzer = "Analyzer instance"
        else:
            pass

        with pytest.raises(UnboundLocalError, match="analyzer"):
            # This would be: del good_residuals, bad_residuals, analyzer
            # We can't use del directly with pytest.raises for multiple targets,
            # so we simulate the failing access:
            del analyzer  # noqa: F821


# ---------------------------------------------------------------------------
# 3. Pattern audit: no other del statements have the same bug
# ---------------------------------------------------------------------------

class TestNoOtherConditionalDelBugs:
    """Scan the entire run() function for other del statements that might
    have the same conditional-assignment bug."""

    def test_all_del_targets_reachable(self, tree, source):
        """For every del statement in run(), verify each target variable
        has at least one assignment that is NOT inside an if/for/while/try block."""
        run_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run":
                run_func = node
                break

        assert run_func is not None

        # Collect all del statements in run()
        del_stmts = []
        for node in ast.walk(run_func):
            if isinstance(node, ast.Delete):
                del_stmts.append(node)

        assert len(del_stmts) > 0, "No del statements found in run()"

        # For each del, check that targets are safely assigned
        for del_stmt in del_stmts:
            for target in del_stmt.targets:
                if not isinstance(target, ast.Name):
                    continue
                var = target.id
                # Check if this variable is assigned unconditionally before the del
                # or is assigned in the same block (like merged_model which is
                # assigned and deleted in the same else branch)
                source_lines = source.splitlines()
                del_line = del_stmt.lineno
                # Simple heuristic: check if there's a `var = ...` at the same or
                # lesser indentation before the del
                del_indent = len(source_lines[del_line - 1]) - len(
                    source_lines[del_line - 1].lstrip()
                )
                found = False
                for i in range(del_line - 1, -1, -1):
                    line = source_lines[i]
                    stripped = line.lstrip()
                    indent = len(line) - len(stripped)
                    if re.match(rf"{var}\s*=", stripped):
                        if indent <= del_indent:
                            found = True
                            break
                        # Also OK if in the same block (e.g. else branch)
                        if indent == del_indent + 4:
                            # Check if there's an if/else that contains both
                            found = True
                            break

                assert found, (
                    f"del target '{var}' at line {del_line} may not be assigned "
                    f"before reaching the del statement"
                )


# ---------------------------------------------------------------------------
# 4. Regression guard: the specific crash scenario from issue #299
# ---------------------------------------------------------------------------

class TestIssue299Regression:
    """Exact reproduction of the issue #299 scenario."""

    def test_default_settings_do_not_require_full_residuals(self, source):
        """By default, print_residual_geometry and plot_residuals are both
        False, which means needs_full_residuals is False and analyzer is
        never created. Verify the code handles this correctly."""
        # Verify the condition
        assert "needs_full_residuals = settings.print_residual_geometry or settings.plot_residuals" in source
        # The fix ensures analyzer = None is set before this branch
        assert re.search(
            r"analyzer\s*=\s*None.*?needs_full_residuals",
            source,
            re.DOTALL,
        ), "analyzer must be initialized to None before the needs_full_residuals check"

    def test_traceback_line_is_fixed(self, source):
        """The traceback in issue #299 points to:
            del good_residuals, bad_residuals, analyzer
        Verify this line still exists and all its targets are safe."""
        lines = source.splitlines()
        del_lines = [
            (i + 1, line)
            for i, line in enumerate(lines)
            if "del good_residuals, bad_residuals, analyzer" in line
        ]
        assert len(del_lines) == 1, f"Expected exactly 1 del line, found {len(del_lines)}"

        del_lineno = del_lines[0][0]

        # All three variables must be initialized before this line
        before = "\n".join(lines[:del_lineno - 1])
        for var in ("good_residuals", "bad_residuals", "analyzer"):
            assert re.search(
                rf"^\s+{var}\s*=\s*None", before, re.MULTILINE
            ), f"{var} not initialized before del at line {del_lineno}"

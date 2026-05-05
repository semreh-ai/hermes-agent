from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import config as hermes_config
from hermes_cli import main as hermes_main


def _cp(cmd, returncode=0, stdout="", stderr=""):
    return SimpleNamespace(args=cmd, returncode=returncode, stdout=stdout, stderr=stderr)


class TestUpdateAutoResolveConfig:
    def test_cli_flag_enables_auto_resolve_without_reading_config(self, monkeypatch):
        def fail_load_config():
            raise AssertionError("load_config should not be called for explicit CLI flag")

        monkeypatch.setattr(hermes_config, "load_config", fail_load_config)

        assert hermes_main._should_auto_resolve_update_conflicts(
            SimpleNamespace(auto_resolve_conflicts=True)
        ) is True

    def test_cli_no_flag_disables_auto_resolve_even_when_config_enabled(self, monkeypatch):
        def fail_load_config():
            raise AssertionError("load_config should not be called for explicit CLI flag")

        monkeypatch.setattr(hermes_config, "load_config", fail_load_config)

        assert hermes_main._should_auto_resolve_update_conflicts(
            SimpleNamespace(auto_resolve_conflicts=False)
        ) is False

    def test_absent_cli_flag_uses_config_value(self, monkeypatch):
        monkeypatch.setattr(
            hermes_config,
            "load_config",
            lambda: {"updates": {"auto_resolve_conflicts": True}},
        )

        assert hermes_main._should_auto_resolve_update_conflicts(SimpleNamespace()) is True

    def test_missing_or_invalid_config_defaults_false(self, monkeypatch):
        monkeypatch.setattr(
            hermes_config,
            "load_config",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        assert hermes_main._should_auto_resolve_update_conflicts(SimpleNamespace()) is False

    def test_default_config_disables_auto_resolve(self):
        assert hermes_config.DEFAULT_CONFIG["updates"]["auto_resolve_conflicts"] is False
        assert "conflict_resolver" in hermes_config.DEFAULT_CONFIG["updates"]


class TestUpdateAutoResolveStagingWorktree:
    def test_clean_staging_merge_verifies_promotes_and_runs_post_steps(
        self, monkeypatch, tmp_path
    ):
        live = tmp_path / "repo"
        live.mkdir()
        staging_seen = []
        recorded = []
        promoted = []
        posts = []

        def fake_run(cmd, **kwargs):
            cwd = Path(kwargs.get("cwd", live))
            recorded.append((cmd, cwd))
            if cmd == ["git", "rev-parse", "HEAD"] and cwd == live:
                return _cp(cmd, stdout="base123\n")
            if cmd == ["git", "status", "--porcelain"] and cwd == live:
                return _cp(cmd, stdout="")
            if cmd[:3] == ["git", "worktree", "add"]:
                staging_seen.append(Path(cmd[-2]))
                return _cp(cmd)
            if cmd == ["git", "merge", "upstream/main", "--no-edit"]:
                assert cwd == staging_seen[0]
                return _cp(cmd, stdout="Merge made by ort.\n")
            if cmd == ["git", "rev-parse", "HEAD"] and cwd == staging_seen[0]:
                return _cp(cmd, stdout="merged456\n")
            if cmd[:3] == ["git", "worktree", "remove"]:
                assert Path(cmd[-1]) == staging_seen[0]
                return _cp(cmd)
            if cmd[:3] == ["git", "branch", "-D"]:
                return _cp(cmd)
            raise AssertionError(f"unexpected command {cmd} cwd={cwd}")

        monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)
        monkeypatch.setattr(hermes_main, "_run_update_verification", lambda *a, **kw: True)
        monkeypatch.setattr(
            hermes_main,
            "_promote_staged_update_to_live",
            lambda *a, **kw: promoted.append((a, kw)) or True,
        )
        monkeypatch.setattr(
            hermes_main,
            "_run_post_update_steps",
            lambda **kw: posts.append(kw),
        )

        result = hermes_main._fork_auto_merge_upstream_via_staging_worktree(
            ["git"],
            live,
            "jonathan/custom-main",
            SimpleNamespace(auto_resolve_conflicts=True),
            gateway_mode=False,
            gw_input_fn=None,
            auto_stash_ref=None,
            prompt_for_restore=False,
        )

        assert result is True
        assert promoted
        assert posts and posts[0]["branch"] == "jonathan/custom-main"
        assert any(cmd[:3] == ["git", "worktree", "add"] for cmd, _ in recorded)
        assert any(cmd[:3] == ["git", "worktree", "remove"] for cmd, _ in recorded)

    def test_conflicted_staging_merge_invokes_resolver_commits_then_promotes(
        self, monkeypatch, tmp_path
    ):
        live = tmp_path / "repo"
        live.mkdir()
        staging_seen = []
        recorded = []
        resolver_calls = []
        promoted = []

        unmerged_sequences = iter([
            ["hermes_cli/main.py"],  # after failed merge
            [],                       # after resolver
        ])

        def fake_run(cmd, **kwargs):
            cwd = Path(kwargs.get("cwd", live))
            recorded.append((cmd, cwd))
            if cmd == ["git", "rev-parse", "HEAD"] and cwd == live:
                return _cp(cmd, stdout="base123\n")
            if cmd == ["git", "status", "--porcelain"] and cwd == live:
                return _cp(cmd, stdout="")
            if cmd[:3] == ["git", "worktree", "add"]:
                staging_seen.append(Path(cmd[-2]))
                return _cp(cmd)
            if cmd == ["git", "merge", "upstream/main", "--no-edit"]:
                return _cp(
                    cmd,
                    returncode=1,
                    stdout="CONFLICT (content): Merge conflict in hermes_cli/main.py\n",
                    stderr="Automatic merge failed; fix conflicts and then commit the result.\n",
                )
            if cmd == ["git", "add", "-A"]:
                return _cp(cmd)
            if cmd == ["git", "commit", "--no-edit"]:
                return _cp(cmd, stdout="[branch abc] merge\n")
            if cmd == ["git", "rev-parse", "HEAD"] and cwd == staging_seen[0]:
                return _cp(cmd, stdout="merged456\n")
            if cmd[:3] == ["git", "worktree", "remove"]:
                return _cp(cmd)
            if cmd[:3] == ["git", "branch", "-D"]:
                return _cp(cmd)
            raise AssertionError(f"unexpected command {cmd} cwd={cwd}")

        def fake_collect(*args, **kwargs):
            return next(unmerged_sequences)

        def fake_resolver(**kwargs):
            resolver_calls.append(kwargs)
            return True

        monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)
        monkeypatch.setattr(hermes_main, "_collect_unmerged_files", fake_collect)
        monkeypatch.setattr(hermes_main, "_invoke_update_conflict_resolver_agent", fake_resolver)
        monkeypatch.setattr(hermes_main, "_run_update_verification", lambda *a, **kw: True)
        monkeypatch.setattr(
            hermes_main,
            "_promote_staged_update_to_live",
            lambda *a, **kw: promoted.append((a, kw)) or True,
        )
        monkeypatch.setattr(hermes_main, "_run_post_update_steps", lambda **kw: None)

        result = hermes_main._fork_auto_merge_upstream_via_staging_worktree(
            ["git"],
            live,
            "jonathan/custom-main",
            SimpleNamespace(auto_resolve_conflicts=True),
            gateway_mode=False,
            gw_input_fn=None,
            auto_stash_ref=None,
            prompt_for_restore=False,
        )

        assert result is True
        assert resolver_calls
        assert resolver_calls[0]["conflicted_files"] == ("hermes_cli/main.py",)
        assert any(cmd == ["git", "commit", "--no-edit"] for cmd, _ in recorded)
        assert promoted

    def test_resolver_failure_keeps_live_repo_unpromoted_and_restores_stash(
        self, monkeypatch, tmp_path, capsys
    ):
        live = tmp_path / "repo"
        live.mkdir()
        staging_seen = []
        promoted = []
        restored = []

        def fake_run(cmd, **kwargs):
            cwd = Path(kwargs.get("cwd", live))
            if cmd == ["git", "rev-parse", "HEAD"] and cwd == live:
                return _cp(cmd, stdout="base123\n")
            if cmd == ["git", "status", "--porcelain"] and cwd == live:
                return _cp(cmd, stdout="")
            if cmd[:3] == ["git", "worktree", "add"]:
                staging_seen.append(Path(cmd[-2]))
                return _cp(cmd)
            if cmd == ["git", "merge", "upstream/main", "--no-edit"]:
                return _cp(cmd, returncode=1, stdout="CONFLICT\n", stderr="")
            raise AssertionError(f"unexpected command {cmd} cwd={cwd}")

        monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)
        monkeypatch.setattr(hermes_main, "_collect_unmerged_files", lambda *a, **kw: ["x.py"])
        monkeypatch.setattr(hermes_main, "_invoke_update_conflict_resolver_agent", lambda **kw: False)
        monkeypatch.setattr(
            hermes_main,
            "_promote_staged_update_to_live",
            lambda *a, **kw: promoted.append((a, kw)) or True,
        )
        monkeypatch.setattr(
            hermes_main,
            "_restore_stashed_changes",
            lambda *a, **kw: restored.append((a, kw)) or True,
        )

        with pytest.raises(SystemExit) as exc:
            hermes_main._fork_auto_merge_upstream_via_staging_worktree(
                ["git"],
                live,
                "jonathan/custom-main",
                SimpleNamespace(auto_resolve_conflicts=True),
                gateway_mode=False,
                gw_input_fn=None,
                auto_stash_ref="stash123",
                prompt_for_restore=False,
            )

        assert exc.value.code == 1
        assert promoted == []
        assert restored
        out = capsys.readouterr().out
        assert "Auto-resolve failed" in out
        assert "staging worktree" in out


class TestResolverAgentInvocation:
    def test_invokes_hermes_chat_in_live_repo_with_staging_terminal_cwd(
        self, monkeypatch, tmp_path
    ):
        live = tmp_path / "live"
        staging = tmp_path / "staging"
        live.mkdir()
        staging.mkdir()
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return _cp(cmd, stdout="RESOLVER_RESULT: success\n")

        monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)
        monkeypatch.setattr(hermes_main, "PROJECT_ROOT", live)
        monkeypatch.setattr(hermes_main.sys, "executable", "/venv/bin/python")

        ok = hermes_main._invoke_update_conflict_resolver_agent(
            git_cmd=["git"],
            live_cwd=live,
            staging_cwd=staging,
            current_branch="jonathan/custom-main",
            base_sha="base123",
            target_ref="upstream/main",
            conflicted_files=("hermes_cli/main.py",),
            args=SimpleNamespace(),
            resolver_cfg={"max_turns": 33, "timeout_seconds": 77, "toolsets": ["terminal", "file"]},
        )

        assert ok is True
        cmd, kwargs = calls[0]
        assert cmd[:5] == ["/venv/bin/python", "-P", "-m", "hermes_cli.main", "chat"]
        assert "--yolo" in cmd
        assert "--accept-hooks" in cmd
        assert "--ignore-rules" in cmd
        assert "--max-turns" in cmd and "33" in cmd
        assert kwargs["cwd"] == staging
        assert kwargs["timeout"] == 77
        assert kwargs["stdin"] is hermes_main.subprocess.DEVNULL
        assert kwargs["env"]["TERMINAL_CWD"] == str(staging)
        assert kwargs["env"]["HERMES_UPDATE_RESOLVER"] == "1"


class TestUpdateAutoResolveRealGit:
    def test_real_git_conflict_resolution_is_staged_before_commit(
        self, monkeypatch, tmp_path
    ):
        repo = tmp_path / "repo"
        subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
        (repo / "hermes_cli").mkdir()
        target = repo / "hermes_cli" / "main.py"
        target.write_text("base\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)

        subprocess.run(["git", "checkout", "-q", "-b", "upstream/main"], cwd=repo, check=True)
        target.write_text("upstream\n")
        subprocess.run(["git", "commit", "-q", "-am", "upstream"], cwd=repo, check=True)

        subprocess.run(["git", "checkout", "-q", "-b", "custom", "main"], cwd=repo, check=True)
        target.write_text("custom\n")
        subprocess.run(["git", "commit", "-q", "-am", "custom"], cwd=repo, check=True)

        def fake_resolver(**kwargs):
            resolved = kwargs["staging_cwd"] / "hermes_cli" / "main.py"
            resolved.write_text("custom\nupstream\n")
            return True

        monkeypatch.setattr(hermes_main, "_invoke_update_conflict_resolver_agent", fake_resolver)
        monkeypatch.setattr(hermes_main, "_run_post_update_steps", lambda **kw: None)
        monkeypatch.setattr(hermes_main, "_clear_bytecode_cache", lambda cwd: 0)
        monkeypatch.setattr(
            hermes_main,
            "_get_update_conflict_resolver_config",
            lambda args: {
                "worktree_parent": str(tmp_path / "worktrees"),
                "keep_failed_worktree": True,
                "max_turns": 5,
                "timeout_seconds": 60,
                "toolsets": ["terminal", "file"],
                "verify_commands": [],
            },
        )

        result = hermes_main._fork_auto_merge_upstream_via_staging_worktree(
            ["git"],
            repo,
            "custom",
            SimpleNamespace(auto_resolve_conflicts=True),
            gateway_mode=False,
            gw_input_fn=None,
            auto_stash_ref=None,
            prompt_for_restore=False,
        )

        assert result is True
        assert target.read_text() == "custom\nupstream\n"
        assert subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ) == ""
        parents = subprocess.check_output(
            ["git", "rev-list", "--parents", "-n", "1", "HEAD"],
            cwd=repo,
            text=True,
        ).split()
        assert len(parents) == 3

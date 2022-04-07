import os
import sys
import subprocess
import errno


def run_command(commands, args, cwd=None, verbose=False, hide_stderr=False):
    assert isinstance(commands, list)
    p = None
    for c in commands:
        try:
            # remember shell=False, so use git.cmd on windows, not just git
            p = subprocess.Popen([c] + args,
                                 cwd=cwd,
                                 stdout=subprocess.PIPE,
                                 stderr=(subprocess.PIPE if hide_stderr else None))
            break
        except EnvironmentError as ex:
            print("EnvironmentError: " + str(ex))
            e = sys.exc_info()[1]
            if e.errno == errno.ENOENT:
                continue
            if verbose:
                print("unable to run %s" % args[0])
                print(e)
            return False, str(ex)
        except Exception as ex:
            print("EnvironmentError: " + str(ex))
            return False, str(ex)

    else:
        if verbose:
            print("unable to find command, tried %s" % (commands,))
        return False, ""
    stdout = p.communicate()[0].strip()
    if sys.version >= '3':
        stdout = stdout.decode()
    if p.returncode != 0:
        if verbose:
            print("unable to run %s (error)" % args[0])
        return False, stdout
    return True, stdout


def git_current_branch(root):
    lines = os.popen("git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \\(.*\\)/ (\\1)/'").readlines()
    line = lines[0].strip()
    line = line.replace("(", "")
    line = line.replace(")", "")
    return line


def git_versions_from_vcs(root, verbose=False):
    # this runs 'git' from the root of the source tree. This only gets called
    # if the git-archive 'subst' keywords were *not* expanded, and
    # _version.py hasn't already been rewritten with a short version string,
    # meaning we're inside a checked out source tree.

    if not os.path.exists(os.path.join(root, ".git")):
        if verbose:
            print("no .git in %s" % root)
        return {}

    GITS = ["git"]
    if sys.platform == "win32":
        GITS = ["git.cmd", "git.exe"]
    rc, stdout = run_command(GITS,
                             ["describe", "--tags", "--dirty", "--always"],
                             cwd=root)
    if not rc:
        return {}
    tag = stdout
    rc, stdout = run_command(GITS,
                             ["rev-parse", "HEAD"],
                             cwd=root)
    if stdout is None:
        return {}
    full = stdout.strip()
    if tag.endswith("-dirty"):
        full += "-dirty"
    branch = git_current_branch(root)
    return {"version": tag, "full": full, "branch": branch}


# def git_versions_oneline(root, verbose=False):
#     if not os.path.exists(os.path.join(root, ".git")):
#         if verbose:
#             print("no .git in %s" % root)
#             return
#     os.system("git --no-pager log --decorate --pretty=oneline --abbrev-commit -1")

def check_git_version(git_ver_str):
    if git_ver_str is None:
        return "gv_empty"

    if "-dirty" in git_ver_str:
        return "gv_dirtry"
    if "-g" in git_ver_str:
        return "gv_not_tagged"
    return "gv_tagged"


if __name__ == '__main__':
    from pprint import pprint
    dir_root = os.path.dirname(os.path.dirname(__file__))
    git_info = git_versions_from_vcs(dir_root)
    pprint(git_info)

    #branch = git_current_branch(dir_root)
    #print(branch)

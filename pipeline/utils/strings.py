import re
from ast import literal_eval


def leanfile_replace_slash(path, repl):
    """
    Replaces the path of a .lean file by removing the .lean suffix and replaces its '/' with another string.
    """
    return path.replace("/", repl)[:-5]


def is_lean_suffix(path):
    """Returns whether the suffix to the given path is '.lean' or not, i.e. if it is a .lean file."""
    return path[-min(5, len(path)):] == ".lean"


def pos_conversion(position_str):
    """Converts a string indiciating the position of a theorem into a tuple of two integers."""
    return literal_eval(position_str.replace('⟨', '(').replace('⟩', ')')) 


def invalid_error_msg(error_msg):
    """Checks if the error message is invalid or not."""
    cond1 = error_msg == ""
    cond2 = "warning:" in error_msg and "error:" not in error_msg
    return cond1 or cond2


def remove_comments(file_str):
    """Remove all comments/annotations from the string."""

    # remove lines that begin with '--' comment
    # regex_pattern = r"(^--.*?\n)"
    # filtered_file_str = re.sub(regex_pattern, "\n", file_str, flags=re.MULTILINE)

    # regex_pattern = r"(/--.*?-/\n)" # remove everything between /-- and -/
    # regex_pattern += r"|(--.*?\n)" # remove content that begin with --
    # regex_pattern += r"|(#align.*?\n)" # remove lines that begin with #align
    # regex_pattern += r"|(/-.*?-/\n)" # remove everything between /-. and -/ 
    # filtered_file_str = re.sub(regex_pattern, "", filtered_file_str, flags=re.DOTALL)
    # filtered_file_str = re.sub(r"^library_note.*\n?", "", filtered_file_str, flags=re.MULTILINE) # remove lines that begin with library_note
    # return filtered_file_str


    # remove wrapped comments that are not followed by a newline    
    # regex_pattern = r'(((/-)|(/--)).*?-/(?!\n))'
    # filtered_file_str = re.sub(regex_pattern, '', file_str, flags=re.DOTALL)
    regex_pattern = r'(((/-)|(/--)).*?-/(?:\n\s*)?)'
    filtered_file_str = re.sub(regex_pattern, '', file_str, flags=re.DOTALL)

    # remove comments that are followed by a newline
    regex_pattern = r'(\n\s*?--[^\n]*)'
    regex_pattern += r'|(--[^\n]*)'
    regex_pattern += r'|(#align.*?\n)'
    filtered_file_str = re.sub(regex_pattern, '', filtered_file_str)
    filtered_file_str = re.sub(r'^library_note.*\n?', '', filtered_file_str, flags=re.MULTILINE) # remove lines that begin with library_note
    filtered_file_str = re.sub(r'^set_option.*\n?', '', filtered_file_str, flags=re.MULTILINE) # remove lines that begin with set_option
    return filtered_file_str


def theorem_name_regex():
    """Returns the regular expression for extracting the theorem name."""
    return re.compile(r"^\s*(?:.*\s)?(?:theorem|lemma)\s+(\S+)", re.MULTILINE)


def theorem_proof_regex(thm_statement):
    """Returns the regular expression for extracting the theorem proof."""
    thm_proof_regex = r'(?<='
    thm_proof_regex += re.escape(thm_statement)
    thm_proof_regex += r')(.*?)(?=\n\n\S|\n\S|$)'
    return thm_proof_regex


def theorem_body_regex(theorem_name):
    """Returns the regular expression for extracting the body of a specific theorem."""
    thm_regex = r"((?:theorem|lemma)\s+"
    thm_regex += re.escape(theorem_name)
    thm_regex += r".*?)"
    thm_regex += r"(?=\n+?\S|$)"
    return thm_regex
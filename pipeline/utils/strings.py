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


def remove_comments(file_str):
    """Remove all comments/annotations from the string."""
    # remove wrapped comments that are not followed by a newline    
    regex_pattern = r'(((/-)|(/--)).*?-/(?:\n\s*)?)'
    filtered_file_str = re.sub(regex_pattern, '', file_str, flags=re.DOTALL)

    # remove comments that are followed by a newline
    regex_pattern = r'(\n\s*?--[^\n]*)'
    regex_pattern += r'|(--[^\n]*)'
    regex_pattern += r'|(#align.*?\n)'
    filtered_file_str = re.sub(regex_pattern, '', filtered_file_str)
    filtered_file_str = re.sub(r'^library_note.*\n?', '', filtered_file_str, flags=re.MULTILINE) # remove lines that begin with library_note
    filtered_file_str = re.sub(r'^set_option.*\n?', '', filtered_file_str, flags=re.MULTILINE) # remove lines that begin with set_option
    filtered_file_str = re.sub(r'^add_decl_doc.*\n?', '', filtered_file_str, flags=re.MULTILINE) # remove lines that begin with #adaptation_note
    filtered_file_str = re.sub(r'^\s*?#adaptation_note\s*?\n', '', filtered_file_str, flags=re.MULTILINE) # remove lines that begin with #adaptation_note
    filtered_file_str = re.sub(r'#adaptation_note\s*', '', filtered_file_str, flags=re.MULTILINE) # remove substrings that contain #adaptation_notes
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
import unidecode

def clean_name(name):
    name = name.replace(" ", "_")
    name = name.replace("-", "_")
    name = unidecode.unidecode(name)
    return name.strip().lower()

def combine_clean_names(first_name, second_name):
    full_name = first_name + '_' + second_name
    return clean_name(full_name)
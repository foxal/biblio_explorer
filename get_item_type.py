# -*- coding: utf-8 -*-
# This script is used to determine the type of an item (person, organization, or something else) based on its name. GPt-4o is used to guess the type, and human confirmation can be requested. If the user does not respond within 5 seconds, the default type is used. The user can also manually specify the type of the item. The script returns the type of the item in the form of a string.
# auto_type_check = 0: manual; 1: with human confirmation; 2: without human confirmation

from inputimeout import inputimeout, TimeoutOccurred
from gpt import GPTResponse

def get_item_type(item_name, auto_type_check=1, gui_mode=False, gui_callback=None):
    '''Determine the type of an item (person, organization, or something else) based on its name. Return the type of the item and whether it was determined automatically. If auto_type_check is 0, the user must manually specify the type. If auto_type_check is 1, GPT is used to guess the type, and human confirmation is requested. If auto_type_check is 2, GPT is used to guess the type, and human confirmation is not requested. If the user does not respond within the designated time-out, the default type is used.'''

    def manual_type_check(manual_item_name):
        if gui_mode and gui_callback:
            # Use GUI callback to get input if in GUI mode
            return gui_callback(manual_item_name)
        else:
            # Use console input if not in GUI mode
            item_is_type = input(f'You decide: is "{manual_item_name}" a person, organization, publication or something else? [P/o/b/e]: ') or "p"
            if item_is_type.lower() == "p":
                return "person"
            elif item_is_type.lower() == "o":
                return "organization"
            elif item_is_type.lower() == "b":
                return "publication"
            else:
                item_type = input(f'Please specify the type for "{manual_item_name}": ')
                return item_type

    suggested_type = None
    if auto_type_check == 1 or auto_type_check == 2:
        gpt_type_judge = GPTResponse()
        prompt_text = f"""Is '{item_name}' a person, an organization, or a publication? If person, return "person", if organization return "organization", if publication, return "publication". No explanation."""
        suggested_type_response = gpt_type_judge.get_response(prompt_text).lower()
        if "person" in suggested_type_response:
            suggested_type = "person"
        elif "organization" in suggested_type_response:
            suggested_type = "organization"
        elif "publication" in suggested_type_response:
            suggested_type = "publication"
        else:
            suggested_type = "unclassified"

        if auto_type_check == 1: # with human confirmation
            # print('Confirmation required. Timeout=5s. y or Enter to agree, n for the other in person or organization, m for manual input.')
            is_auto = False
            
            # Handle confirmation with GUI or console
            if gui_mode and gui_callback:
                confirmation = gui_callback(f'GPT suggests "{item_name}" is a(n) {suggested_type}. Do you agree?')
            else:
                try:
                    confirmation = inputimeout(prompt=f'GPT suggests "{item_name}" is a(n) {suggested_type}. Do you agree? [Y/n/m]: ', timeout=20) or "y"
                except TimeoutOccurred:
                    print("Time is out. Defaulting to GPT suggestion.")
                    confirmation = "y"
                    is_auto = True
                    
            # Process confirmation result
            if confirmation.lower() == 'y':
                item_type = suggested_type
            elif confirmation.lower() == 'n' and suggested_type in ["person", "organization"]:
                item_type = "organization" if suggested_type == "person" else "person"
            elif confirmation.lower() == 'm':
                item_type = manual_type_check(item_name)
            else:
                item_type = "unclassified"
        elif auto_type_check == 2: # without human confirmation
            is_auto = True
            if suggested_type in ["person", "organization", "publication"]:
                item_type = suggested_type
            else:
                item_type = "unclassified"
    else:
        is_auto = False
        item_type = manual_type_check(item_name)

    return item_type, is_auto

if __name__ == "__main__":
    item_name = "未来学へのおすすめ"

    # use GPT to guess item type?
    auto_type_check = 1 # 0: manual; 1: yes, with human confirmation; 2: yes, without human confirmation

    print(get_item_type(item_name, auto_type_check))
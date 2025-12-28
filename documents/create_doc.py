import json
from utils import ndex_to_ids, clean_text, form_pokedex_entry, create_type_matchups, create_evolution_paths

# --- LOAD DATA ---
with open('pokemon_forms.json', 'r', encoding='utf-8') as f:
    pokemon_forms = json.load(f)
with open('ndex_names.json', 'r', encoding='utf-8') as f:
    national_pokedex = json.load(f)
with open('regions.json', 'r', encoding='utf-8') as f:
    regions = json.load(f)
with open('types.json', 'r', encoding='utf-8') as f:
    types = json.load(f)
with open('evolutions.json', 'r', encoding='utf-8') as f:
    evolutions = json.load(f)
with open('abilities.json', 'r', encoding='utf-8') as f:
    abilities = json.load(f)
with open('type_weakness_charts.json', 'r', encoding='utf-8') as f:
    type_weakness_chart = json.load(f)

# --- LOOKUP TABLES ---
ndex_id_to_form_ids = ndex_to_ids(pokemon_forms)
type_matchups = create_type_matchups(type_weakness_chart)
evolution_map = create_evolution_paths(evolutions)
evolution_text = {e['pokemon_form_id']: clean_text(e['text_markup']) for e in evolutions}
abilities = {a['id']: clean_text(a['name']) for a in abilities}
form_ids = {f['id']: f for f in pokemon_forms}
types = {t['id']: t['name'] for t in types}
national_pokedex = {f['ndex_id']: f['name_english'] for f in national_pokedex}


with open('_pokedex_knowledge_base.txt', 'w', encoding='utf-8') as file:
    for ndex_id in ndex_id_to_form_ids.keys():
        name = national_pokedex[ndex_id]

        # Pokedex data, Base stats, Type defenses for each form
        entry_groups = form_pokedex_entry(ndex_id_to_form_ids[ndex_id], types, abilities, type_matchups, evolution_map) # {entry_text: [form, ..., form]}

        n_forms = len(ndex_id_to_form_ids[ndex_id])
        for entry_text in entry_groups.keys():
            form_names = []
            for form in entry_groups[entry_text]:
                # Form name
                if form["form_name"] != "Default Form" and n_forms > 1:
                    form_names.append(f"{name} ({form['form_name']})")
                else:
                    form_names.append(name)

                # Evolution
                has_evolution = False
                evo_lines = evolution_map.get(form['id'])
                if evo_lines:
                    evo_text = []
                    for evo_line in evo_lines:
                        if len(evo_line) > 1:
                            has_evolution = True
                            evo_texts = []
                            for id in evo_line:
                                f_data = form_ids[id]
                                cur_name = national_pokedex[f_data["ndex_id"]]
                                cur_n_forms = len(ndex_id_to_form_ids[f_data["ndex_id"]])
                                
                                # Formatting the name based on form/default status
                                name_str = f"{cur_name} ({f_data['form_name']})" if f_data['form_name'] != "Default Form" and cur_n_forms > 1 else cur_name
                                condition = evolution_text.get(id)

                                if condition:
                                    evo_texts.append(f"({condition}) {name_str}")
                                else:
                                    evo_texts.append(name_str)
    
                            evo_text.append(f" -> ".join(evo_texts))
                
            if not has_evolution:
                evo_text = "This Pokemon does not evolve."
            else:
                evo_text = "\n".join(evo_text)
            form_names = ", ".join(form_names)
            file.write(f"###{form_names}:\n{entry_text}\nEvolution:\n{evo_text}\n")


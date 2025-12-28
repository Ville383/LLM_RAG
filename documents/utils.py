from collections import defaultdict
import re


def ndex_to_ids(pokemon_forms: list):
    '''Create a lookup table from national pokedex id to included pokemon form ids.'''
    ndex_id_to_form_id = defaultdict(list)
    for form in pokemon_forms:
        if form['include'] and not form['is_mega_evolution'] and not form['is_gmax_form']:
            ndex_id_to_form_id[form['ndex_id']].append(form)
    return ndex_id_to_form_id


def create_type_matchups(type_weakness_chart):
    type_matchups = {}
    for row in type_weakness_chart:
        key = (row["defense_type_1_id"], row["defense_type_2_id"])
        type_matchups[key] = {k: float(v) for k, v in row.items() if k not in {"id", "defense_type_1_id", "defense_type_2_id"}}
    return type_matchups


def create_evolution_paths(evolutions):
    """
    Parse evolution data and return all evolution paths.
    
    Returns:
        dict: {pokemon_form_id: [[path1], [path2], ...]}
    """
    # Group entries by their tree ID
    trees = defaultdict(list)
    for entry in evolutions:
        # Clean the grid data: handle "1 / span 2" or strings
        col = int(str(entry['grid_column']).split('/')[0].strip())
        row = int(str(entry['grid_row']).split('/')[0].strip())
        
        entry['_c'] = col
        entry['_r'] = row
        trees[entry['evolution_tree_id']].append(entry)

    final_map = defaultdict(list)

    for _, members in trees.items():
        # Build an adjacency list: parent_id -> [child_ids]
        adj = defaultdict(list)
        
        # Sort by column to ensure we process parents before children
        members.sort(key=lambda x: x['_c'])
        
        columns = defaultdict(list)
        for m in members:
            columns[m['_c']].append(m)
            
        # Determine relationships
        for m in members:
            parent_col_idx = m['_c'] - 1
            if parent_col_idx in columns:
                potential_parents = columns[parent_col_idx]
                
                # Match by row index
                parent = next((p for p in potential_parents if p['_r'] == m['_r']), None)
                
                # If no exact row match, the first entry in the previous column is the parent
                # (Common for branching base forms like Eevee or Tyrogue)
                if not parent:
                    parent = potential_parents[0]
                
                adj[parent['pokemon_form_id']].append(m['pokemon_form_id'])

        # 2. Find all unique paths from Column 1 (roots) to the leaves
        roots = [m['pokemon_form_id'] for m in members if m['_c'] == 1]
        all_paths = []

        def find_paths_dfs(current_id, current_path):
            children = adj.get(current_id, [])
            if not children:
                all_paths.append(current_path)
                return
            for child in children:
                find_paths_dfs(child, current_path + [child])

        for root_id in roots:
            find_paths_dfs(root_id, [root_id])

        # 3. Map every Pokémon in this tree to every path they participate in
        tree_pokemon_ids = {m['pokemon_form_id'] for m in members}
        for p_id in tree_pokemon_ids:
            for path in all_paths:
                if p_id in path:
                    final_map[p_id].append(path)

    return dict(final_map)



def clean_text(text: str):
    if text is None:
        return ""
    
    text = re.sub(r'<[^>]+>', ' ', text)

    def replace_braces(match):
        return match.group(1).split(';')[0]
    
    text = re.sub(r'\{([^}]+)\}', replace_braces, text)
    return text.replace('\n', ' ').replace('  ', ' ').strip()


def format_types(types):
    return ", ".join(t.capitalize() for t in sorted(types))


def form_pokedex_entry(ndex_forms: list, types: dict, abilities: dict, type_matchups: dict, evolution_map: dict):
    entries = defaultdict(list)
    for form in ndex_forms:
        # Info
        entry_text = f"\nPokédex data\nNational No: {form['ndex_id']}\nSpecies: {form['pokemon_category']}\nHeight: {form['height_m']}\nWeight: {form['weight_kg']}\n"

        # Typing
        type1, type2 = types.get(form['type_1_id']), types.get(form['type_2_id'])
        entry_text += "Type: "
        entry_text += f"{type1}/{type2}\n" if type2 else f"{type1}\n"

        # Abilities
        entry_text += "Abilities:\n"
        ability_ids = [form["ability_primary_id"], form["ability_secondary_id"], form["ability_hidden_id"],]
        if all(aid is None for aid in ability_ids):
            entry_text += "    —\n"
        else:
            for lbl, aid in [("1.", form["ability_primary_id"]), ("2.", form["ability_secondary_id"]), ("Hidden:", form["ability_hidden_id"]),]:
                if aid is not None:
                    entry_text += f"    {lbl} {abilities[aid]}\n"

        # Base stats
        entry_text += "\nBase stats:\n"
        entry_text += f"| HP: {form['stat_hp']} | Atk: {form['stat_attack']} | Def: {form['stat_defense']} | Sp.Atk: {form['stat_spatk']} | Sp.Def: {form['stat_spdef']} | Spd: {form['stat_speed']} | Total: {form['stat_total']} |\n\n"
        
        # Type matchup
        matchup = type_matchups.get((form['type_1_id'], form['type_2_id']), {})
        categories = {"immune": [], "quad_resist": [], "resist": [], "weak": [], "quad_weak": []}
        for atk_type, mult in matchup.items():
            if mult == 0: categories["immune"].append(atk_type)
            elif mult == 0.25: categories["quad_resist"].append(atk_type)
            elif mult == 0.5: categories["resist"].append(atk_type)
            elif mult == 2: categories["weak"].append(atk_type)
            elif mult == 4: categories["quad_weak"].append(atk_type)

        entry_text += "Type defenses:\n"
        for k, v in [("Immune (0x)", "immune"), ("Strongly Resists (0.25x)", "quad_resist"), ("Resists (0.5x)", "resist"), ("Weak (2x)", "weak"), ("Strongly Weak (4x)", "quad_weak")]:
            if categories[v]:
               entry_text += f"{k}: {format_types(categories[v])}\n"
        
        entries[entry_text].append(form)
        
    return entries
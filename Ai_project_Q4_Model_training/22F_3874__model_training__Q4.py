from collections import defaultdict

states = ['WA', 'NT', 'Q', 'NSW', 'V', 'SA', 'T']
domains = {state: ['red', 'green', 'blue'] for state in states}

neighbors = {
    'WA': ['NT', 'SA'],
    'NT': ['WA', 'Q', 'SA'],
    'Q': ['NT', 'NSW', 'SA'],
    'NSW': ['Q', 'V', 'SA'],
    'V': ['NSW', 'SA'],
    'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
    'T': []
}

def is_consistent(state, color, assignment):
    for neighbor in neighbors[state]:
        if neighbor in assignment and assignment[neighbor] == color:
            return False
    return True

def mrv(assignment):
    unassigned = [v for v in states if v not in assignment]
    return min(unassigned, key=lambda var: len(domains[var]))

def degree(var):
    return len(neighbors[var])
def lcv(var):
    return sorted(domains[var], key=
                  val: sum(
        1 for neighbor in neighbors[var] if val in domains[neighbor]
    ))

# Forward Checking
def forward_check(assignment, var, value):
    inferences = []
    for neighbor in neighbors[var]:
        if neighbor not in assignment and value in domains[neighbor]:
            domains[neighbor].remove(value)
            inferences.append((neighbor, value))
            if not domains[neighbor]:
                return False, []
    return True, inferences

# Backtracking algorithm
def backtrack(assignment):
    if len(assignment) == len(states):
        return assignment
    var = mrv(assignment)
    for value in lcv(var):
        if is_consistent(var, value, assignment):
            assignment[var] = value
            success, inferences = forward_check(assignment, var, value)
            if success:
                result = backtrack(assignment)
                if result:
                    return result
            del assignment[var]
            for (neighbor, val) in inferences:
                domains[neighbor].append(val)
    return None

solution = backtrack({})
print("Solution:", solution)
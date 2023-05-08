def min_edit_distance(s1, s2):
    # Create a table to store the minimum edit distance between prefixes of s1 and s2
    table = [[0 for j in range(len(s2)+1)] for i in range(len(s1)+1)]

    # Initialize the table with base cases
    for i in range(1, len(s1)+1):
        table[i][0] = i
    for j in range(1, len(s2)+1):
        table[0][j] = j

    # Compute the minimum edit distance for all prefixes of s1 and s2
    for i in range(1, len(s1)+1):
        for j in range(1, len(s2)+1):
            # Compute the cost of each operation
            cost_replace = 0 if s1[i-1] == s2[j-1] else 1
            cost_insert = 1
            cost_delete = 1

            # Choose the operation that minimizes the edit distance
            table[i][j] = min(table[i-1][j-1]+cost_replace, table[i-1][j]+cost_delete, table[i][j-1]+cost_insert)

    # Return the minimum edit distance
    return table[len(s1)][len(s2)]

# testing
print(min_edit_distance("cat", "adventure"))
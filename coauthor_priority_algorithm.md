# Coauthor Priority Value Calculation Algorithm

This document explains how the priority value for coauthors is calculated in the Biblio Explorer's network expansion algorithm. The priority value determines the order in which authors are processed during the network expansion.

## Overview

The priority value is a number between 0 and 1 that combines two factors:
1. **Depth Priority**: Based on how far the coauthor is from the seed author
2. **Closeness Priority**: Based on the coauthor's relationship to prioritized authors

These two factors are combined using a configurable weight parameter to produce the final priority value.

## Detailed Algorithm

### Input Parameters

- `coauthors`: List of coauthor data (id, name, type, publication count, depth)
- `prioritized_authors`: List of author IDs that are considered high priority
- `prioritized_depth_weight`: A value between 0 and 1 that determines how much weight to give to closeness vs. depth (default: 0.8)

### Step 1: Calculate Depth Priority

For each coauthor, the depth priority is calculated as:

```
depth_priority = 0.9 ^ coauthor_depth
```

This means:
- Depth 0 (seed author): 0.9⁰ = 1.0
- Depth 1: 0.9¹ = 0.9
- Depth 2: 0.9² = 0.81
- And so on...

The deeper the coauthor is in the network, the lower their depth priority.

### Step 2: Calculate Closeness Priority

The closeness priority measures how closely connected a coauthor is to the prioritized authors. It's calculated as follows:

1. If the coauthor is directly in the prioritized set:
   ```
   closeness_priority = 1.0
   ```

2. Otherwise, calculate based on network connections:
   - Find the coauthor's 1st degree neighbors (direct co-authors)
   - Find the coauthor's 2nd degree neighbors (co-authors of co-authors)
   - Calculate the overlap between these neighbors and the prioritized set:
     ```
     overlap_1st = number of 1st degree neighbors in prioritized set
     overlap_2nd = number of 2nd degree neighbors in prioritized set
     ```
   - Apply weights to these overlaps:
     ```
     weight_1st = 1.0
     weight_2nd = 0.5
     raw_closeness_score = (weight_1st * overlap_1st) + (weight_2nd * overlap_2nd)
     ```
   - Normalize the score relative to the size of the prioritized set:
     ```
     max_possible_score = size of prioritized set * weight_1st
     closeness_priority = raw_closeness_score / max_possible_score
     ```
   - Ensure the closeness priority doesn't exceed 1.0

### Step 3: Combine Priorities

The final priority is a weighted combination of depth priority and closeness priority:

```
combined_priority = (1 - prioritized_depth_weight) * depth_priority + prioritized_depth_weight * closeness_priority
```

If prioritization is not active (no prioritized authors or prioritized_depth is set to "off"), then:

```
combined_priority = depth_priority
```

The final priority value is clamped to the range [0, 1].

## Example Calculation

Consider a coauthor at depth 2 with:
- 3 first-degree neighbors in the prioritized set
- 5 second-degree neighbors in the prioritized set
- Total prioritized set size of 10 authors
- prioritized_depth_weight = 0.8

1. Depth priority = 0.9² = 0.81
2. Closeness priority:
   - raw_closeness_score = (1.0 * 3) + (0.5 * 5) = 3 + 2.5 = 5.5
   - max_possible_score = 10 * 1.0 = 10
   - closeness_priority = 5.5 / 10 = 0.55
3. Combined priority = (1 - 0.8) * 0.81 + 0.8 * 0.55 = 0.162 + 0.44 = 0.602

This coauthor would have a priority value of 0.602.

## Significance

The priority value determines which authors are processed first during network expansion:
- Higher priority values (closer to 1.0) mean the author will be processed earlier
- The algorithm balances breadth-first search (by depth) with targeted exploration (by closeness to prioritized authors)
- By adjusting the prioritized_depth_weight, users can control whether the expansion focuses more on depth (breadth-first) or closeness to prioritized authors

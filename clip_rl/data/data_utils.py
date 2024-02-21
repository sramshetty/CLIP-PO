
# Extending https://github.com/Sohojoe/CLIP_visual-spatial-reasoning/blob/main/src/data000.py
NEGATE = {
    # Adjacency
    "adjacent to": "nonadjacent to", 
    "alongside": "away from", 
    "at the side of": "away from", 
    "at the right side of": "at the left side of", 
    "at the left side of": "at the right side of",
    "attached to": "disconnect from", 
    "at the back of": "at the front of", 
    "ahead of": "not ahead of", 
    "against": "away from", 
    "at the edge of": "far from the edge of", 
    # Directional
    "off": "on", 
    "past": "before", 
    "toward": "away from", 
    "down": "up", 
    "away from": "not away from", 
    "along": "not along", 
    "around": "not around", 
    "into": "not into", 
    "across": "not accross",
    "across from": "not across from",
    "through": "not through",  
    "down from": "up from",
    # Orientation
    "facing": "facing away from", 
    "facing away from": "facing", 
    "parallel to": "perpendicular to", 
    "perpendicular to": "parallel to", 
    # Proximity
    "by": "far away from", 
    "close to": "far from", 
    "near": "far from", 
    "far from": "close to", 
    "far away from": "by", 
    # Topological
    "connected to": "detached from", 
    "detached from": "connected to", 
    "has as a part": "does not have a part", 
    "part of": "not part of", 
    "contains": "does not contain", 
    "within": "outside of", 
    "at": "not at", 
    "on": "not on", 
    "in": "not in",
    "with": "not with", 
    "surrounding": "not surrounding", 
    "among": "not among", 
    "consists of": "does not consists of", 
    "out of": "not out of", 
    "between": "not between", 
    "inside": "outside", 
    "outside": "inside", 
    "touching": "not touching",
    # Unallocated
    "beyond": "inside",
    "next to": "far from", 
    "opposite to": "not opposite to", 
    "enclosed by": "not enclosed by", 
    # missing
    "above": "below",
    "below": "above",
    "behind": "in front of",
    "on top of": "not on top of",
    "under": "over",
    "over": "under",
    "left of": "right of",
    "right of": "left of",
    "in front of": "behind",
    "beneath": "not beneath",
    "beside": "not beside",
    "in the middle of": "not in the middle of",
    "congruent": "incongruent",
}
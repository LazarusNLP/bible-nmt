# Missing Verses Analysis Report

## Overview

This analysis compared 51 English Bible versions against the Dhao (nfa) Bible to identify missing verses where content exists in one language but not the other.

## Key Statistics

- **Total verses checked**: 41,899
- **English versions analyzed**: 51
- **Dhao content distribution**:
  - Content verses: 9,072 (21.6%)
  - Empty verses: 32,410 (77.4%)
  - Range markers: 417 (1.0%)

## Overall Mismatch Summary

Across all English versions:
- **English has content, Dhao missing**: 884,310 mismatches
- **Dhao has content, English missing**: 103,965 mismatches  
- **English content, Dhao range**: 16,394 mismatches
- **Dhao content, English range**: 806 mismatches

## Best Aligned English Versions (Fewest Mismatches)

1. **engtnt**: 1,728 mismatches
   - 0 English content/Dhao missing
   - 1,424 Dhao content/English missing
   - 304 English content/Dhao range

2. **engemtv**: 1,731 mismatches  
   - 0 English content/Dhao missing
   - 1,427 Dhao content/English missing
   - 304 English content/Dhao range

3. **engf35**: 1,732 mismatches
   - 0 English content/Dhao missing
   - 1,428 Dhao content/English missing
   - 304 English content/Dhao range

4. **engtcent**: 1,732 mismatches
   - 0 English content/Dhao missing
   - 1,428 Dhao content/English missing  
   - 304 English content/Dhao range

5. **engPEV**: 3,829 mismatches
   - 772 English content/Dhao missing
   - 2,472 Dhao content/English missing
   - 226 English content/Dhao range
   - 359 Dhao content/English range

## Worst Aligned English Versions (Most Mismatches)

1. **uk_lxx2012**: 33,448 mismatches
2. **lxx2012**: 33,448 mismatches  
3. **Brenton**: 33,054 mismatches
4. **englxxup**: 32,676 mismatches
5. **engjps**: 29,245 mismatches

## Common Patterns

### New Testament Only Versions
Several English versions (engtnt, engemtv, engf35, engtcent) appear to contain only New Testament content, which explains why they have 0 cases of "English content/Dhao missing" but many cases of "Dhao content/English missing" (Old Testament verses).

### Old Testament Differences  
Many mismatches occur in Old Testament books, particularly:
- 1 Chronicles (1CH)
- Various Psalms
- Deuterocanonical books (present in some versions but not others)

### Range Handling Issues
There are systematic differences in how verse ranges are handled:
- English versions have content where Dhao has range markers
- Less commonly, Dhao has content where English has range markers

## Recommendations

1. **For best alignment with Dhao Bible**: Use engtnt, engemtv, engf35, or engtcent (all have very similar patterns)

2. **For complete Bible coverage**: These New Testament-only versions won't work for Old Testament alignment

3. **For balanced coverage**: Consider engPEV which has relatively few mismatches while maintaining broader coverage

4. **Avoid for alignment**: The LXX-based versions (lxx2012, uk_lxx2012, Brenton) have the most mismatches

## Files Generated

1. `missing_verses_detailed.csv` - Complete mismatch details (1M+ rows)
2. `missing_verses_summary.csv` - Summary by English version
3. `analysis_summary.md` - This report

## Technical Notes

The analysis properly handles:
- Verse ranges marked with `<range>` tokens
- Empty verses (present in structure but no content)
- Different biblical canons (Protestant vs. Catholic vs. Orthodox)
- Line-by-line alignment using verse reference file

Missing verses represent actual content differences between Bible versions, not alignment errors.

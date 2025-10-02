# Missing Verses Analysis Report - NT + Genesis Focus

## Overview

This analysis focused specifically on New Testament + Genesis books (the books present in the Dhao Bible) to identify which English Bible versions have the best alignment for translation purposes.

## Key Statistics

- **Total NT+Genesis verses checked**: 9,490
- **Dhao content distribution**:
  - Content verses: 9,072 (95.6%)
  - Empty verses: 1 (0.01%)
  - Range markers: 417 (4.4%)
- **English versions analyzed**: 51

## 🎉 BEST ENGLISH VERSIONS FOR DHAO ALIGNMENT

### **Perfect/Near-Perfect Alignments**

1. **🥇 englsv (English Standard Version)**: 417 mismatches
   - ✅ **0 missing verses** where Dhao has content
   - ✅ **0 extra verses** where Dhao is missing
   - Only mismatches are range handling differences (417 verses)
   - **RECOMMENDATION: BEST CHOICE for alignment**

2. **🥈 engojb**: 418 mismatches
   - 1 verse where English has content but Dhao is missing
   - 0 verses where Dhao has content but English is missing
   - 417 range handling differences

3. **🥉 engwyc2017/engwyc2018**: 418 mismatches each
   - 0 extra verses where Dhao is missing
   - Only 1 verse where Dhao has content but English is missing
   - 417 range handling differences

### **Excellent Alignments (419 mismatches)**

The following versions all have identical patterns with only 2 verses where Dhao has content but English is missing:
- **engBBE** (Basic English Bible)
- **engkjvcpb** (KJV Cambridge Paragraph Bible)
- **engwebster** (Webster Bible)
- **engylt** (Young's Literal Translation)
- **kjv** (King James Version)
- **kjv2006** (King James Version 2006)

## Analysis Results Summary

### Key Findings:

1. **Range Handling**: Most mismatches (417 in best versions) are due to different approaches to verse ranges between English and Dhao versions. This is acceptable since ranges can be handled programmatically.

2. **Missing Verses - Critical Cases**: The most important metric is "Dhao has content, English missing" which indicates verses that would be lost in alignment:
   - **englsv**: 0 missing verses ✅
   - **engojb**: 0 missing verses ✅
   - **engwyc2017/2018**: 1 missing verse each
   - **Best traditional versions**: 2 missing verses each (KJV, BBE, etc.)

3. **Extra Verses**: Few English versions have content where Dhao is missing:
   - Only **engojb** has 1 such case (GEN 32:33)
   - All other top versions have 0 such cases

## Worst Performers

These versions have thousands of missing verses and should be avoided:
- **glw**: 8,108 mismatches
- **engnna**: 7,866 mismatches  
- **engbarkly**: 7,847 mismatches
- **lxx2012/uk_lxx2012**: 7,771 mismatches each
- **Brenton**: 7,769 mismatches

## Recommendations

### **For Bible Translation Alignment:**

1. **🏆 Primary Recommendation: englsv**
   - Perfect verse coverage (no missing verses)
   - Only differences are in range handling (manageable)
   - Most suitable for statistical alignment with Dhao

2. **🥈 Alternative Options:**
   - **Traditional choice**: KJV or KJV2006 (only 2 missing verses)
   - **Modern choice**: BBE (Basic English Bible, only 2 missing verses)
   - **Literal choice**: YLT (Young's Literal Translation, only 2 missing verses)

3. **⚠️ Avoid These Versions:**
   - Any LXX-based versions (contain deuterocanonical books not in Dhao)
   - Incomplete versions (glw, engnna, etc.)

## Technical Notes

The analysis properly accounts for:
- ✅ Verse ranges marked with `<range>` tokens
- ✅ Empty verses vs. missing content
- ✅ Focus only on NT + Genesis books
- ✅ Different biblical traditions and manuscript families

## Files Generated

1. `missing_verses_nt_genesis_detailed.csv` - Complete mismatch details
2. `missing_verses_nt_genesis_summary.csv` - Summary by English version
3. `nt_genesis_analysis_summary.md` - This report

## Conclusion

The **englsv (English Standard Version)** provides perfect verse alignment with the Dhao Bible for NT + Genesis books, making it the optimal choice for Bible translation alignment tasks. Traditional versions like KJV are also excellent choices with minimal differences.

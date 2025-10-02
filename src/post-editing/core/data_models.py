"""
Data models for post-editing pipeline.
"""

import csv
import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

# Import prompt handling from parent directory
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from prompt import get_prompt


class GlossaryEntry(BaseModel):
    """Data model for a glossary entry."""
    
    source_word: str
    target_word: str
    pos_tag: Optional[str] = None


class Row(BaseModel):
    """Data model for a single translation row."""
    
    src_text: str
    tgt_text: str
    pred_tgt_text: str = Field(description="Original NLLB model prediction from input CSV")
    post_edited_tgt_txt: Optional[str] = None  # For post-editing mode
    translated_tgt_txt: Optional[str] = None   # For translation mode
    src_lang: str = Field(description="Src lang alpha-2 code")
    tgt_lang: str = Field(description="Tgt lang alpha-2 code")
    src_lang_name: str = Field(description="Source language name")
    tgt_lang_name: str = Field(description="Target language name")
    
    # Individual metric improvements (post-edited - original)
    spbleu_improvement: Optional[float] = None
    chrf3_improvement: Optional[float] = None
    chrfpp_improvement: Optional[float] = None

    @classmethod
    def get_savepath(cls, output_dir: str, model: str, csv_path: str, src: str, tgt: str) -> str:
        """Generate output filepath for results."""
        csv_name = csv_path.split('/')[-1].replace('.csv', '')
        return f"{output_dir}/{model.split('/')[-1]}_{csv_name}_{src}_{tgt}.csv"

    def get_messages(self, prompt_key: str = "default", few_shot_examples: List[tuple] = None, 
                     glossary_entries: List[GlossaryEntry] = None, translation_mode: bool = False) -> List[dict]:
        """Generate chat messages for this row."""
        # Get the prompt template
        prompt_template = get_prompt(prompt_key)
        
        # Build the prompt content using the template
        system_content = prompt_template["system"]
        
        # Use the user_template from prompt with proper variable substitution
        if translation_mode:
            # For translation mode, don't include pred_text
            base_user_content = prompt_template["user_template"].format(
                src_lang_name=self.src_lang_name,
                src_text=self.src_text,
                tgt_lang_name=self.tgt_lang_name
            )
        else:
            # For post-editing mode, include pred_text
            base_user_content = prompt_template["user_template"].format(
                src_lang_name=self.src_lang_name,
                src_text=self.src_text,
                tgt_lang_name=self.tgt_lang_name,
                pred_text=self.pred_tgt_text
            )
        
        # Add few-shot examples if provided
        if few_shot_examples:
            examples_text = f"\n\nTo help with the translation, here are some example parallel sentences between {self.tgt_lang_name} and {self.src_lang_name}:\n"
            for src_ex, tgt_ex in few_shot_examples:
                examples_text += f"{self.tgt_lang_name}: {tgt_ex}\n{self.src_lang_name} translation: {src_ex}\n\n"
            base_user_content += examples_text
        
        # Add glossary entries if provided
        if glossary_entries:
            glossary_text = f"\n\nTo help with the translation, here is a word list bewteen {self.src_lang_name} and {self.tgt_lang_name} in the format: English word (pos tag) -> Dhao word:\n"
            for entry in glossary_entries:
                if entry.pos_tag:
                    glossary_text += f"- {entry.source_word} ({entry.pos_tag}) -> {entry.target_word}\n"
                else:
                    glossary_text += f"- {entry.source_word} -> {entry.target_word}\n"
            base_user_content += glossary_text
        
        user_content = base_user_content
        
        # Return messages with system and user content
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        return messages

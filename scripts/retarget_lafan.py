#!/usr/bin/env python3
"""Run genuine native UMR using new Geno/OpenSprite surface correspondence."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from retarget.umr_pipeline import main
if __name__=='__main__':main()

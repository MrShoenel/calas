from pathlib import Path
from sys import path


NOTEBOOKS_DIR = Path(__file__).parent.resolve()
CALAS_DIR = NOTEBOOKS_DIR.parent.joinpath('./calas').resolve()
ROOT_DIR = CALAS_DIR.parent.resolve()

path.append(str(ROOT_DIR))

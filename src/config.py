"""Configuration et initialisation du projet.

Charge config.yaml et configure le logging.
Point d'entrée : setup_project() pour tout initialiser.
"""

import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Charge le fichier config.yaml.

    Parameters
    ----------
    config_path : Path
        Chemin vers le fichier de configuration.

    Returns
    -------
    dict
        Configuration complète du projet.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: dict) -> None:
    """Configure le logging à partir de la config.

    Parameters
    ----------
    config : dict
        Configuration complète (section `logging` utilisée).
    """
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("file", "outputs/project.log")

    # Créer le dossier si nécessaire
    log_path = PROJECT_ROOT / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def setup_project() -> dict:
    """Initialise le projet : .env, config, logging, dossiers.

    Returns
    -------
    dict
        Configuration complète du projet.
    """
    # Charger les variables d'environnement
    load_dotenv(PROJECT_ROOT / ".env")

    # Charger la config
    config = load_config()

    # Configurer le logging
    setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info("Projet initialisé : %s", config["project"]["name"])

    # Créer les dossiers de sortie
    for subdir in ["maps", "figures", "reports"]:
        (PROJECT_ROOT / "outputs" / subdir).mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

    return config


def get_column_name(config: dict, feature_name: str) -> str:
    """Résout un nom de feature projet vers le nom réel BDNB.

    Parameters
    ----------
    config : dict
        Configuration complète.
    feature_name : str
        Nom de la feature dans le projet (ex: 'surface_bat').

    Returns
    -------
    str
        Nom de la colonne réelle dans les données.

    Raises
    ------
    ValueError
        Si le mapping n'est pas défini ou vaut 'TODO'.
    """
    mapping = config.get("features", {}).get("column_mapping", {})
    real_name = mapping.get(feature_name)

    if real_name is None:
        raise ValueError(
            f"Feature '{feature_name}' absente du column_mapping dans config.yaml"
        )
    if real_name == "TODO":
        raise ValueError(
            f"Feature '{feature_name}' a le mapping 'TODO'. "
            f"Compléter config.yaml après exploration (notebook 01)."
        )
    return real_name


if __name__ == "__main__":
    cfg = setup_project()
    print(f"Config chargée : {len(cfg)} sections")
    print(f"Zone : {cfg['zone']['nom']}")
    print(f"Communes : {len(cfg['zone']['codes_insee'])}")

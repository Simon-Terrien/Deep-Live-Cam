#!/usr/bin/env python3

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import time
from pathlib import Path

# Définition des niveaux de log
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO, 
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

# Couleurs pour la console
class ColoredFormatter(logging.Formatter):
    """
    Formatter personnalisé qui ajoute des couleurs aux logs console
    """
    COLORS = {
        "DEBUG": "\033[94m",     # Bleu
        "INFO": "\033[92m",      # Vert
        "WARNING": "\033[93m",   # Jaune
        "ERROR": "\033[91m",     # Rouge
        "CRITICAL": "\033[1;91m" # Rouge gras
    }
    RESET = "\033[0m"

    def format(self, record):
        log_message = super().format(record)
        if record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{log_message}{self.RESET}"
        return log_message

class Logger:
    """
    Classe singleton pour la gestion centralisée des logs
    """
    _instance = None
    _initialized = False
    _logger = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self, name="DeepLiveCam", level="info", log_file=None):
        if not self._initialized:
            # Créer un logger avec le nom spécifié
            self._logger = logging.getLogger(name)
            self._logger.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))
            self._logger.propagate = False
            
            # Formateur par défaut
            formatter = logging.Formatter(
                "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            
            # Création du répertoire logs s'il n'existe pas
            if log_file:
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
            else:
                # Répertoire logs par défaut
                log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                log_file = os.path.join(log_dir, f"{name.lower()}_{time.strftime('%Y%m%d')}.log")
            
            # Handler pour la sortie fichier
            file_handler = RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
            
            # Handler pour la sortie console avec couleurs
            console_handler = logging.StreamHandler(sys.stdout)
            colored_formatter = ColoredFormatter(
                "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S"
            )
            console_handler.setFormatter(colored_formatter)
            self._logger.addHandler(console_handler)
            
            self._initialized = True
            self._logger.debug(f"Logger initialized with level {level}")

    def get_logger(self):
        """Retourne l'instance de logger"""
        return self._logger
    
    def set_level(self, level):
        """Change le niveau de log"""
        if level.lower() in LOG_LEVELS:
            self._logger.setLevel(LOG_LEVELS[level.lower()])
            self._logger.debug(f"Log level changed to {level}")
        else:
            self._logger.warning(f"Unknown log level: {level}")

# Fonctions d'aide pour l'accès rapide aux méthodes de log
def get_logger(name="DeepLiveCam", level=None):
    """Obtient un logger pour le module spécifié"""
    logger_instance = Logger(name, level=level if level else "info")
    return logger_instance.get_logger()

def set_global_level(level):
    """Définit le niveau de log global"""
    logger_instance = Logger()
    logger_instance.set_level(level)
    
# Usage examples:
# logger = get_logger(__name__)
# logger.debug("This is a debug message")
# logger.info("This is an info message")
# logger.warning("This is a warning")
# logger.error("This is an error")
# logger.critical("This is a critical error")
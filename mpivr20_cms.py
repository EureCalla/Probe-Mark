"""Central path settings for Probe-Mark."""

PROBE_MARK_DATA_DIR = r"\\mpi-file01\VPCRD2-AI\08_Training\Probe-Mark\data"
PROBE_MARK_DB_PATH = rf"{PROBE_MARK_DATA_DIR}\probe_mark.db"


def get_clean_output_dir() -> str:
    """Return the shared output directory for cleaned training data."""
    return PROBE_MARK_DATA_DIR


def get_database_path() -> str:
    """Return the shared SQLite database path."""
    return PROBE_MARK_DB_PATH

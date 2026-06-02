"""Central path settings for Probe-Mark."""

PROBE_MARK_ROOT_DIR = r"\\mpi-file01\VPCRD2-AI\08_Training\Probe-Mark"
PROBE_MARK_DATA_DIR = rf"{PROBE_MARK_ROOT_DIR}\data"
PROBE_MARK_DB_PATH = rf"{PROBE_MARK_ROOT_DIR}\probe_mark.db"


def get_output_root_dir() -> str:
    """Return the shared Probe-Mark output root directory."""
    return PROBE_MARK_ROOT_DIR


def get_clean_output_dir() -> str:
    """Return the shared data directory for cleaned training images."""
    return PROBE_MARK_DATA_DIR


def get_database_path() -> str:
    """Return the shared SQLite database path."""
    return PROBE_MARK_DB_PATH

import structlog

logger = structlog.get_logger()

def main() -> None:
    """Minimal entry point for the document intelligence refinery."""
    logger.info("Initializing document-intelligence-refinery...")
    print("Welcome to Document Intelligence Refinery!")

if __name__ == "__main__":
    main()

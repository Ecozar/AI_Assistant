import logging
from AI_Project_Brain.db_manager import db_manager
from AI_Project_Brain.auto_tagger import auto_tagger

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_auto_tagging():
    """Test the improved auto-tagging system"""
    
    # 1. First add some test tags
    test_tags = {
        "python": "programming language coding development",
        "ai": "artificial intelligence machine learning",
        "testing": "verification validation quality assurance",
        "database": "storage data sql queries"
    }
    
    logger.info("Adding test tags...")
    with db_manager.get_cursor() as cursor:
        for tag, description in test_tags.items():
            cursor.execute("""
                INSERT OR REPLACE INTO tags (name, description)
                VALUES (?, ?)
            """, (tag, description))
    
    # 2. Test tag manager refresh
    approved_tags = auto_tagger.available_tags
    logger.info(f"Loaded approved tags: {list(approved_tags.keys())}")
    
    # 3. Test auto-tagger suggestions
    test_texts = [
        "Here's some Python code for database queries",
        "Working on AI testing procedures",
        "Just a normal message without relevant tags",
    ]
    
    for text in test_texts:
        logger.info(f"\nTesting text: {text}")
        suggestions = auto_tagger.get_suggested_tags(text)
        logger.info(f"Suggested tags with confidence:")
        for tag, confidence in suggestions:
            logger.info(f"- {tag}: {confidence:.2f}")

if __name__ == "__main__":
    test_auto_tagging() 
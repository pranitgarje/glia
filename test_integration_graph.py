import sys
import unittest
from datetime import datetime, timezone

# Adjust this path if necessary to find your 'glia' package
sys.path.insert(0, "/home/claude") 

from neo4j import GraphDatabase
from glia.adapters.graph import GraphDBAdapter

class TestGraphDBAdapterIntegration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Connect to the real Neo4j instance before any tests run."""
        # TODO: Update these credentials to match your Neo4j instance
        cls.uri = "neo4j+s://465e1856.databases.neo4j.io" 
        cls.user = "465e1856"
        cls.password = "z_xnChP1_qlKF14RtlJ6ou_PyoDgKkdw22R4D5uB3zA"
        
        cls.driver = GraphDatabase.driver(cls.uri, auth=(cls.user, cls.password))

    @classmethod
    def tearDownClass(cls):
        """Close the driver connection after all tests finish."""
        cls.driver.close()

    def setUp(self):
        """Wipe old test data and seed new data before EACH test."""
        with self.driver.session() as session:
            # Delete any existing TestDocument nodes to ensure a clean slate
            session.run("MATCH (n:TestDocument) DETACH DELETE n")
            
            # Insert some dummy records with timestamps
            session.run("""
                CREATE (:TestDocument {doc_id: 'real-doc-1', last_modified: datetime('2024-01-01T10:00:00Z')})
            """)
            session.run("""
                CREATE (:TestDocument {doc_id: 'real-doc-2', last_modified: datetime('2024-01-01T12:00:00Z')})
            """)

    def tearDown(self):
        """Clean up the database after EACH test."""
        with self.driver.session() as session:
            session.run("MATCH (n:TestDocument) DETACH DELETE n")

    def test_real_database_polling(self):
        """Test that the adapter successfully polls the real database."""
        
        # We use a cursor slightly older than our inserted data
        initial_cursor = datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
        
        adapter = GraphDBAdapter(
            driver=self.driver,
            change_query=None,       # Fix is here!
            mode="polling",
            source_id_field="doc_id",
            node_label="TestDocument", 
            database=None,
            poll_interval=10.0,
            last_cursor=initial_cursor,
        )

        # 1. Test Connection
        adapter.connect()
        self.assertTrue(adapter._connected)

        # 2. Test Polling
        records = list(adapter.poll())
        
        # We expect 2 records based on our setUp data
        self.assertEqual(len(records), 2)
        
        # 3. Test ID Mapping
        source_ids = [adapter.map_to_source_id(rec) for rec in records]
        self.assertIn("node:TestDocument|id:real-doc-1", source_ids)
        self.assertIn("node:TestDocument|id:real-doc-2", source_ids)

        # 4. Test Cursor Advancement
        adapter.advance_cursor(adapter.get_cursor())
        self.assertGreater(adapter.last_cursor, initial_cursor)


if __name__ == "__main__":
    unittest.main(verbosity=2)
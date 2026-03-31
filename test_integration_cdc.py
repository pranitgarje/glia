import sys
import time
import threading
import unittest

# Adjust this path if necessary to find your 'glia' package
sys.path.insert(0, "/home/claude") 

from neo4j import GraphDatabase
from glia.adapters.graph import GraphDBAdapter

class TestGraphDBAdapterCDCIntegration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Connect to the real Neo4j instance before any tests run."""
        cls.uri = "neo4j+s://465e1856.databases.neo4j.io" 
        cls.user = "465e1856"
        cls.password = "z_xnChP1_qlKF14RtlJ6ou_PyoDgKkdw22R4D5uB3zA"
        
        cls.driver = GraphDatabase.driver(cls.uri, auth=(cls.user, cls.password))

    @classmethod
    def tearDownClass(cls):
        """Close the driver connection after all tests finish."""
        cls.driver.close()

    def setUp(self):
        """Wipe old test data to ensure a clean slate."""
        with self.driver.session() as session:
            session.run("MATCH (n:TestCDCDocument) DETACH DELETE n")

    def tearDown(self):
        """Clean up the database after the test."""
        with self.driver.session() as session:
            session.run("MATCH (n:TestCDCDocument) DETACH DELETE n")

    def test_real_database_cdc_stream(self):
        """Test that the adapter successfully streams CDC events."""
        
        # 1. Initialize the adapter with all our hard-learned fixes
        adapter = GraphDBAdapter(
            driver=self.driver,
            change_query=None,       # Required positional argument
            mode="cdc",              # Switched to CDC mode
            source_id_field="doc_id",
            node_label="TestCDCDocument", 
            database=None,           # Use default database routing
            poll_interval=1.0,       # Fast interval so the built-in CDC query loops quickly
        )

        adapter.connect()
        self.assertTrue(adapter._connected)

        # We will store the WatcherEvents yielded by the thread here
        received_events = []
        
        def consume_events():
            """Worker function to consume the blocking generator."""
            try:
                for event in adapter.listen():
                    received_events.append(event)
            except Exception as e:
                print(f"CDC stream exited with error: {e}")

        # 2. Start listening in a background thread
        cdc_thread = threading.Thread(target=consume_events, daemon=True)
        cdc_thread.start()

        # Give the adapter a moment to call db.cdc.earliest() and bootstrap its cursor
        time.sleep(2)

        # 3. Trigger a mutation in the database
        with self.driver.session() as session:
            session.run("CREATE (:TestCDCDocument {doc_id: 'cdc-real-1'})")

        # 4. Wait for the adapter's polling loop to catch the event
        time.sleep(3)

        # 5. Signal the generator to stop cleanly and wait for the thread to exit
        adapter.stop()
        cdc_thread.join(timeout=5)

        # 6. Assertions!
        self.assertGreater(len(received_events), 0, "No CDC events were captured!")
        
        # Find the specific event for our created node
        creation_event = next(
            (e for e in received_events if e.source_id == "node:TestCDCDocument|id:cdc-real-1"), 
            None
        )
        
        self.assertIsNotNone(creation_event, "Did not receive a CDC event for 'cdc-real-1'")
        
        # Validate the payload mappings defined in graph.py
        self.assertEqual(creation_event.payload["operation"], "created")
        self.assertEqual(creation_event.payload["fs_event_type"], "modified")

if __name__ == "__main__":
    unittest.main(verbosity=2)
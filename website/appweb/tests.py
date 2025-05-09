from django.test import TestCase, SimpleTestCase

# Create your tests here.
class SimpleTests(SimpleTestCase):
    def test_appweb_status(self):
        respond = self.client.get('/')
        self.assertEqual(respond.status_code,200)
        
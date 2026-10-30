import os
import tempfile
import types
import unittest
import zipfile
from unittest.mock import MagicMock, patch

from ua_tts.stanza_cache import (
    find_cached_stanza_dir,
    is_stanza_model_cached,
    patch_stanza_cache,
)


class StanzaCacheTests(unittest.TestCase):
    def test_find_cached_stanza_dir_via_env(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res_json = os.path.join(tmp_dir, "resources.json")
            uk_dir = os.path.join(tmp_dir, "uk")
            os.makedirs(uk_dir, exist_ok=True)
            with open(res_json, "w", encoding="utf-8") as f:
                f.write("{}")

            with patch.dict(os.environ, {"STANZA_RESOURCES_DIR": tmp_dir}):
                found = find_cached_stanza_dir("uk")
                self.assertEqual(found, os.path.abspath(tmp_dir))

    def test_find_cached_stanza_dir_none_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.dict(os.environ, {"STANZA_RESOURCES_DIR": tmp_dir, "LOCALAPPDATA": tmp_dir}):
                found = find_cached_stanza_dir("uk")
                self.assertNotEqual(found, tmp_dir)

    def test_is_stanza_model_cached_true_when_files_present(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res_json = os.path.join(tmp_dir, "resources.json")
            tok_dir = os.path.join(tmp_dir, "uk", "tokenize")
            os.makedirs(tok_dir, exist_ok=True)
            with open(res_json, "w", encoding="utf-8") as f:
                f.write("{}")
            with open(os.path.join(tok_dir, "iu.pt"), "w", encoding="utf-8") as f:
                f.write("model-weights")

            self.assertTrue(is_stanza_model_cached(tmp_dir, "uk"))

    def test_is_stanza_model_cached_extracts_zip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res_json = os.path.join(tmp_dir, "resources.json")
            with open(res_json, "w", encoding="utf-8") as f:
                f.write("{}")

            uk_dir = os.path.join(tmp_dir, "uk")
            os.makedirs(uk_dir, exist_ok=True)
            zip_path = os.path.join(uk_dir, "default.zip")

            with zipfile.ZipFile(zip_path, "w") as z:
                z.writestr("tokenize/iu.pt", b"token-weights")

            self.assertTrue(is_stanza_model_cached(tmp_dir, "uk"))
            self.assertTrue(os.path.isfile(os.path.join(uk_dir, "tokenize", "iu.pt")))

    def test_is_stanza_model_cached_false_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertFalse(is_stanza_model_cached(tmp_dir, "uk"))

    def test_patch_stanza_cache_skips_download_when_cached(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res_json = os.path.join(tmp_dir, "resources.json")
            tok_dir = os.path.join(tmp_dir, "uk", "tokenize")
            os.makedirs(tok_dir, exist_ok=True)
            with open(res_json, "w", encoding="utf-8") as f:
                f.write("{}")
            with open(os.path.join(tok_dir, "iu.pt"), "w", encoding="utf-8") as f:
                f.write("weights")

            import ua_tts.stanza_cache as sc
            sc._PATCHED = False

            orig_download = MagicMock()
            orig_pipeline_init_calls = []

            class FakePipeline:
                def __init__(self, lang="en", *args, **kwargs):
                    orig_pipeline_init_calls.append((lang, kwargs))

            class FakeDownloadMethod:
                NONE = "NONE"

            fake_core = types.ModuleType("stanza.pipeline.core")
            fake_core.Pipeline = FakePipeline
            fake_core.DownloadMethod = FakeDownloadMethod

            fake_pipeline = types.ModuleType("stanza.pipeline")
            fake_pipeline.core = fake_core

            mock_stanza = types.ModuleType("stanza")
            mock_stanza.download = orig_download
            mock_stanza.Pipeline = FakePipeline
            mock_stanza.pipeline = fake_pipeline

            fake_rc = types.ModuleType("stanza.resources.common")
            fake_rc.DEFAULT_MODEL_DIR = tmp_dir
            fake_resources = types.ModuleType("stanza.resources")
            fake_resources.common = fake_rc
            mock_stanza.resources = fake_resources

            with patch.dict("sys.modules", {
                "stanza": mock_stanza,
                "stanza.pipeline": fake_pipeline,
                "stanza.pipeline.core": fake_core,
                "stanza.resources": fake_resources,
                "stanza.resources.common": fake_rc,
            }):
                result = patch_stanza_cache(preferred_dir=tmp_dir)
                self.assertTrue(result)

                # Test patched download: should NOT call orig_download
                mock_stanza.download("uk", dir=tmp_dir)
                orig_download.assert_not_called()

                # Test pipeline init: should set download_method to NONE
                inst = mock_stanza.Pipeline("uk", dir=tmp_dir)
                self.assertEqual(len(orig_pipeline_init_calls), 1)
                self.assertEqual(orig_pipeline_init_calls[0][1].get("download_method"), "NONE")


if __name__ == "__main__":
    unittest.main()

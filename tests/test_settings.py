import json
from dataclasses import asdict

from sqlite_rag.settings import Settings, SettingsManager


class TestSettings:
    def test_store_settings(self, db_conn):
        settings_manager = SettingsManager(db_conn[0])
        settings = Settings(
            model_path_or_name="test_model",
            model_config="test_config",
            embedding_dim=768,
            vector_type="test_store",
            chunk_overlap=100,
            chunk_size=1000,
            quantize_scan=True,
        )

        settings_manager.store(settings)

        stored_settings = settings_manager.load_settings()

        assert stored_settings is not None
        assert stored_settings.model_path_or_name == "test_model"
        assert stored_settings.model_config == "test_config"
        assert stored_settings.embedding_dim == 768
        assert stored_settings.vector_type == "test_store"
        assert stored_settings.chunk_overlap == 100
        assert stored_settings.chunk_size == 1000
        assert stored_settings.quantize_scan is True

    def test_store_settings_when_exist(self, db_conn):
        settings_manager = SettingsManager(db_conn[0])
        settings = Settings(
            model_path_or_name="test_model",
            model_config="test_config",
            embedding_dim=768,
            vector_type="test_store",
            chunk_overlap=100,
            chunk_size=1000,
            quantize_scan=True,
        )

        settings_manager.store(settings)

        # Store again with different values
        new_settings = Settings(
            model_path_or_name="new_model",
            model_config="new_config",
            embedding_dim=512,
            vector_type="new_store",
            chunk_overlap=50,
            chunk_size=500,
            quantize_scan=False,
        )
        settings_manager.store(new_settings)

        stored_settings = settings_manager.load_settings()

        assert stored_settings is not None
        assert stored_settings.model_path_or_name == "new_model"
        assert stored_settings.model_config == "new_config"
        assert stored_settings.embedding_dim == 512
        assert stored_settings.vector_type == "new_store"
        assert stored_settings.chunk_overlap == 50
        assert stored_settings.chunk_size == 500
        assert stored_settings.quantize_scan is False

    def test_load_settings_when_not_exist(self, db_conn):
        settings_manager = SettingsManager(db_conn[0])
        stored_settings = settings_manager.load_settings()

        assert stored_settings is None

    def test_load_settings_with_defaults(self, db_conn):
        settings_manager = SettingsManager(db_conn[0])
        settings = Settings()
        settings_manager.store(settings)

        loaded_settings = settings_manager.load_settings()

        assert loaded_settings is not None
        assert loaded_settings.model_path_or_name == settings.model_path_or_name
        assert loaded_settings.model_config == settings.model_config
        assert loaded_settings.embedding_dim == settings.embedding_dim
        assert loaded_settings.vector_type == settings.vector_type
        assert loaded_settings.chunk_overlap == settings.chunk_overlap
        assert loaded_settings.chunk_size == settings.chunk_size
        assert loaded_settings.quantize_scan == settings.quantize_scan

    def test_load_settings_when_a_new_property_is_added_to_settinigs(self, db_conn):
        settings_manager = SettingsManager(db_conn[0])
        settings = Settings()

        # Store settings without quantize_scan to simulate a
        # new property being added later
        settings_dict = asdict(settings)
        del settings_dict["quantize_scan"]
        db_conn[0].execute(
            """
            INSERT INTO settings (id, settings)
            VALUES ('1', ?)
            ;
        """,
            (json.dumps(asdict(settings)),),
        )

        loaded_settings = settings_manager.load_settings()

        assert loaded_settings is not None
        assert loaded_settings.quantize_scan  # Default value should be used

    def test_has_critical_changes(self, db_conn):
        settings_manager = SettingsManager(db_conn[0])

        current_settings = Settings()

        has_changes = settings_manager.has_critical_changes(
            current_settings, current_settings
        )
        assert not has_changes

        new_settings = Settings()
        new_settings.model_path_or_name = "modified_model"

        has_changes = settings_manager.has_critical_changes(
            new_settings, current_settings
        )
        assert has_changes

        new_settings = Settings()
        new_settings.embedding_dim = 14

        has_changes = settings_manager.has_critical_changes(
            new_settings, current_settings
        )
        assert has_changes

        new_settings = Settings()
        new_settings.vector_type = "my_changed_vector"

        has_changes = settings_manager.has_critical_changes(
            new_settings, current_settings
        )
        assert has_changes

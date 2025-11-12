import inspect
from dataclasses import fields
from typing import Any, Optional, Union, get_args, get_origin

import typer

from .settings import Settings

SETTINGS_FIELDS = tuple(fields(Settings))
SETTINGS_FIELD_NAMES = {field.name for field in SETTINGS_FIELDS}


def _strip_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _is_bool(annotation: Any) -> bool:
    return _strip_optional(annotation) is bool


def _option_help(field_obj) -> str:
    return field_obj.metadata.get(
        "help", f"Override {field_obj.name.replace('_', ' ')}"
    )


def _cli_name(field_obj) -> str:
    return field_obj.metadata.get("cli_name", field_obj.name.replace("_", "-"))


def _build_setting_parameter(field_obj) -> inspect.Parameter:
    option_names = []
    cli_name = _cli_name(field_obj)
    if _is_bool(field_obj.type):
        option_names.append(f"--{cli_name}/--no-{cli_name}")
    else:
        option_names.append(f"--{cli_name}")

    option = typer.Option(
        None,
        *option_names,
        help=_option_help(field_obj),
        show_default=False,
    )

    annotation = Optional[_strip_optional(field_obj.type)]
    return inspect.Parameter(
        field_obj.name,
        inspect.Parameter.KEYWORD_ONLY,
        default=option,
        annotation=annotation,
    )


def build_configure_signature() -> inspect.Signature:
    params = [
        inspect.Parameter(
            "ctx",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=typer.Context,
        ),
        inspect.Parameter(
            "force",
            inspect.Parameter.KEYWORD_ONLY,
            default=typer.Option(
                False,
                "-f",
                "--force",
                help=(
                    "Force update even if critical settings change "
                    "(like model or embedding dimension)"
                ),
            ),
            annotation=bool,
        ),
    ]

    for field_obj in SETTINGS_FIELDS:
        params.append(_build_setting_parameter(field_obj))

    return inspect.Signature(params)


def filter_setting_updates(settings_values: dict[str, Any]) -> dict[str, Any]:
    """Return only the settings provided as CLI overrides."""
    return {
        key: value
        for key, value in settings_values.items()
        if key in SETTINGS_FIELD_NAMES and value is not None
    }

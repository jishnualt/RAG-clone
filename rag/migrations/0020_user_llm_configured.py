from django.db import migrations, models


def set_llm_configured_for_ready_users(apps, schema_editor):
    User = apps.get_model("rag", "User")
    db_alias = schema_editor.connection.alias
    User.objects.using(db_alias).filter(active_collection__isnull=False, active_collection_ready=True).update(
        llm_configured=True
    )


def noop_reverse(apps, schema_editor):
    # No-op reverse; leaving llm_configured flags as-is.
    return


class Migration(migrations.Migration):

    dependencies = [
        ("rag", "0019_multi_tenant_collections"),
    ]

    operations = [
        migrations.AddField(
            model_name="user",
            name="llm_configured",
            field=models.BooleanField(default=False, help_text="Indicates whether the user has configured an LLM and collection."),
        ),
        migrations.RunPython(set_llm_configured_for_ready_users, noop_reverse),
    ]

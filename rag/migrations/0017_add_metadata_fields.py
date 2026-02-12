from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("rag", "0016_user_collection_created"),
    ]

    operations = [
        migrations.AddField(
            model_name="assistant",
            name="metadata",
            field=models.JSONField(
                blank=True,
                default=dict,
                help_text="Optional key/value metadata for this assistant (max 16 entries).",
            ),
        ),
        migrations.AddField(
            model_name="document",
            name="metadata",
            field=models.JSONField(
                blank=True,
                default=dict,
                help_text="Optional key/value metadata for this document (max 16 entries).",
            ),
        ),
        migrations.AddField(
            model_name="message",
            name="metadata",
            field=models.JSONField(
                blank=True,
                default=dict,
                help_text="Optional key/value metadata for this message (max 16 entries).",
            ),
        ),
        migrations.AddField(
            model_name="run",
            name="metadata",
            field=models.JSONField(
                blank=True,
                default=dict,
                help_text="Optional key/value metadata for this run (max 16 entries).",
            ),
        ),
        migrations.AddField(
            model_name="thread",
            name="metadata",
            field=models.JSONField(
                blank=True,
                default=dict,
                help_text="Optional key/value metadata for this thread (max 16 entries).",
            ),
        ),
        migrations.AddField(
            model_name="vectorstore",
            name="metadata",
            field=models.JSONField(
                blank=True,
                default=dict,
                help_text="Optional key/value metadata for this vector store (max 16 entries).",
            ),
        ),
    ]

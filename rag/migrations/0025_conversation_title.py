from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("rag", "0024_create_conversation_models"),
    ]

    operations = [
        migrations.AddField(
            model_name="conversation",
            name="title",
            field=models.CharField(
                max_length=255,
                blank=True,
                null=True,
                help_text="Optional title for the conversation, typically derived from the first user message.",
            ),
        ),
    ]

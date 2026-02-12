from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("rag", "0020_user_llm_configured"),
    ]

    operations = [
        migrations.AlterField(
            model_name="user",
            name="selected_llm_provider",
            field=models.CharField(blank=True, choices=[("OpenAI", "OpenAI"), ("Ollama", "Ollama"), ("Claude", "Claude")], default=None, help_text="The LLM provider currently selected for this user.", max_length=50, null=True),
        ),
    ]

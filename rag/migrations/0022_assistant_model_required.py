from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("rag", "0021_allow_null_selected_provider"),
    ]

    operations = [
        migrations.AlterField(
            model_name="assistant",
            name="model",
            field=models.CharField(help_text="LLM model to use for this assistant.", max_length=100),
        ),
    ]

from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [
        ('rag', '0009_assistant_model_assistant_tools_run_required_action_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='run',
            name='mode',
            field=models.CharField(choices=[('document', 'Document'), ('normal', 'Normal'), ('web', 'Web')], default='document', max_length=20),
        ),
    ]

# Generated manually

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rag', '0022_assistant_model_required'),
    ]

    operations = [
        migrations.AlterField(
            model_name='thread',
            name='vector_store',
            field=models.ForeignKey(blank=True, null=True, on_delete=models.CASCADE, related_name='threads', to='rag.vectorstore'),
        ),
    ]


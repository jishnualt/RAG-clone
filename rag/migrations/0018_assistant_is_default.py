from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("rag", "0017_add_metadata_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="assistant",
            name="is_default",
            field=models.BooleanField(
                default=False,
                help_text="Marks this assistant as the tenant's default when no assistant is specified for a run.",
            ),
        ),
        migrations.AddConstraint(
            model_name="assistant",
            constraint=models.UniqueConstraint(
                fields=["tenant", "is_default"],
                condition=models.Q(is_default=True),
                name="unique_default_assistant_per_tenant",
            ),
        ),
    ]

from django.db import migrations, models
import django.db.models.deletion
import uuid
from django.utils.text import slugify


def generate_qdrant_name(tenant_id, name: str) -> str:
    slug = slugify(name) or "collection"
    return f"t_{tenant_id}__c_{slug}__{uuid.uuid4().hex[:8]}"


def bootstrap_collections(apps, schema_editor):
    Tenant = apps.get_model("rag", "Tenant")
    User = apps.get_model("rag", "User")
    Collection = apps.get_model("rag", "Collection")
    VectorStore = apps.get_model("rag", "VectorStore")

    db_alias = schema_editor.connection.alias

    for tenant in Tenant.objects.using(db_alias).all():
        legacy_name = getattr(tenant, "collection_name", None) or "default_collection"
        owner = (
            User.objects.using(db_alias)
            .filter(tenant=tenant)
            .order_by("id")
            .first()
        )
        collection = Collection.objects.using(db_alias).create(
            tenant=tenant,
            name=legacy_name,
            owner=owner,
            is_active=True,
            qdrant_collection_name=generate_qdrant_name(tenant.id, legacy_name),
        )
        User.objects.using(db_alias).filter(tenant=tenant).update(
            active_collection=collection,
            active_collection_ready=True,
            is_setup=True,
        )
        VectorStore.objects.using(db_alias).filter(tenant=tenant).update(
            collection=collection
        )


def reverse_bootstrap(apps, schema_editor):
    Collection = apps.get_model("rag", "Collection")
    Collection.objects.all().delete()


class Migration(migrations.Migration):

    dependencies = [
        ("rag", "0018_assistant_is_default"),
    ]

    operations = [
        migrations.CreateModel(
            name="Collection",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("name", models.CharField(max_length=255)),
                ("is_active", models.BooleanField(default=True)),
                ("qdrant_collection_name", models.CharField(max_length=255, unique=True)),
                ("embedding_dimension", models.IntegerField(blank=True, null=True)),
                ("provider", models.CharField(blank=True, choices=[("OpenAI", "OpenAI"), ("Ollama", "Ollama"), ("Claude", "Claude")], max_length=50, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("owner", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="collections", to="rag.user")),
                ("tenant", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="collections", to="rag.tenant")),
            ],
            options={
                "constraints": [],
            },
        ),
        migrations.AddField(
            model_name="user",
            name="active_collection",
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="active_users", to="rag.collection"),
        ),
        migrations.AddField(
            model_name="user",
            name="active_collection_ready",
            field=models.BooleanField(default=False, help_text="Indicates whether the active collection has been provisioned for this user."),
        ),
        migrations.AddField(
            model_name="vectorstore",
            name="collection",
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name="vector_stores", to="rag.collection"),
        ),
        migrations.RunPython(bootstrap_collections, reverse_bootstrap),
        migrations.RemoveField(
            model_name="tenant",
            name="collection_name",
        ),
        migrations.RemoveField(
            model_name="user",
            name="collection_created",
        ),
        migrations.AddConstraint(
            model_name="collection",
            constraint=models.UniqueConstraint(fields=("tenant", "name"), name="unique_collection_name_per_tenant"),
        ),
        migrations.AddConstraint(
            model_name="collection",
            constraint=models.UniqueConstraint(condition=models.Q(is_active=True), fields=("owner",), name="unique_active_collection_per_user"),
        ),
    ]
